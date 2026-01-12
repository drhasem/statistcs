import pandas as pd
import numpy as np
from scipy import stats

# Monkey patch sklearn.utils.check_array to fix 'force_all_finite' issue in factor_analyzer/pingouin
try:
    import sklearn.utils
    import sklearn.utils.validation
    
    # Define patch function
    def _patched_check_array(*args, **kwargs):
        if 'force_all_finite' in kwargs:
            kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
        return _original_check_array(*args, **kwargs)

    # Patch sklearn.utils.validation.check_array
    if hasattr(sklearn.utils.validation, 'check_array'):
        _original_check_array = sklearn.utils.validation.check_array
        sklearn.utils.validation.check_array = _patched_check_array
    
    # Patch sklearn.utils.check_array (often aliased here)
    if hasattr(sklearn.utils, 'check_array'):
        sklearn.utils.check_array = _patched_check_array

except ImportError:
    pass

def clean_numeric_data(df, columns):
    """
    Ensure specific columns are strictly numeric and finite.
    Coerce errors and drop NaNs.
    """
    sub_df = df[columns].copy()
    for col in columns:
        sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
    
    clean_df = sub_df.dropna()
    if clean_df.empty:
        raise ValueError("No valid numeric data remaining after cleaning (check for non-numeric values).")
    return clean_df


def get_descriptive_stats(df, selected_vars=None):
    """
    Calculate descriptive statistics for numerical and categorical variables.
    """
    if selected_vars:
        df = df[selected_vars]
        
    # Numerical Stats
    numeric_df = df.select_dtypes(include=[np.number])
    stats_dict = []
    
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        desc['skew'] = numeric_df.skew()
        desc['kurtosis'] = numeric_df.kurtosis()
        desc = desc.reset_index().rename(columns={'index': 'variable'})
        
        # Calculate Normality Tests (Shapiro & K-S)
        normality_data = []
        for index, row in desc.iterrows():
            var_name = row['variable']
            # Get valid data
            data = numeric_df[var_name].dropna()
            
            # Shapiro-Wilk
            # Note: Shapiro is reliable for N < 5000. For larger N, p-value may be inaccurate but we run it anyway.
            try:
                stat_sh, p_sh = stats.shapiro(data)
                shapiro_res = f"{stat_sh:.3f} (p={p_sh:.3f})"
                shapiro_norm = "Normal" if p_sh > 0.05 else "Not Normal"
            except:
                shapiro_res = "Error"
                shapiro_norm = "-"

            # Kolmogorov-Smirnov
            # We compare against standard normal, so we standardize data first
            try:
                standardized_data = (data - data.mean()) / data.std()
                stat_ks, p_ks = stats.kstest(standardized_data, 'norm')
                ks_res = f"{stat_ks:.3f} (p={p_ks:.3f})"
                ks_norm = "Normal" if p_ks > 0.05 else "Not Normal"
            except:
                ks_res = "Error"
                ks_norm = "-"
                
            normality_data.append({
                'variable': var_name,
                'Shapiro-Wilk': shapiro_res,
                'Shapiro Result': shapiro_norm,
                'K-S Test': ks_res,
                'K-S Result': ks_norm
            })
            
        # Merge normality data
        desc = pd.merge(desc, pd.DataFrame(normality_data), on='variable')
        stats_dict = desc.replace({np.nan: None}).to_dict(orient='records')

    # Categorical Stats (Frequencies)
    categorical_df = df.select_dtypes(exclude=[np.number])
    freq_dict = {}
    
    if not categorical_df.empty:
        for col in categorical_df.columns:
            # Value counts with percentage
            vc = categorical_df[col].value_counts(dropna=False).reset_index()
            vc.columns = ['value', 'count']
            vc['percentage'] = (vc['count'] / len(categorical_df) * 100).round(2)
            # Convert to list of dicts
            freq_dict[col] = vc.replace({np.nan: 'Missing'}).to_dict(orient='records')
            
    # Generate Plots (Histograms for numeric, Bar for categorical)
    plots = {}
    
    # Simple Histogram data for numeric
    for col in numeric_df.columns:
        # Dropna for plotting
        vals = numeric_df[col].dropna().tolist()
        plots[col] = {
            'type': 'histogram',
            'x': vals,
            'name': col,
            'marker': {'color': '#6366f1'}
        }
        
    # Bar Chart data for categorical
    for col in freq_dict:
        data = freq_dict[col]
        plots[col] = {
            'type': 'bar',
            'x': [str(d['value']) for d in data],
            'y': [d['count'] for d in data],
            'name': col,
            'marker': {'color': '#10b981'}
        }
            
    return {
        'numeric': stats_dict,
        'categorical': freq_dict,
        'plots': plots
    }

def calculate_group_differences(df, test_type, group_var, dep_vars):
    """
    Calculate group differences based on the test type.
    Supported types: t_test, mann_whitney, anova, kruskal, paired_t, wilcoxon
    """
    results = []
    
    # Ensure dep_vars is a list
    if isinstance(dep_vars, str):
        dep_vars = [dep_vars]
        
    import pingouin as pg
    from scipy import stats

    for dv in dep_vars:
        res = {}
        res['variable'] = dv
        try:
            # Independent T-Test (2 groups)
            if test_type == 't_test':
                groups = df[group_var].astype(str).unique()
                if len(groups) != 2:
                    res['error'] = f"Requires exactly 2 groups, found {len(groups)}"
                else:
                    gua = df[df[group_var].astype(str) == groups[0]][dv].dropna()
                    gub = df[df[group_var].astype(str) == groups[1]][dv].dropna()
                    
                    # Levene (Equal var)
                    lev = stats.levene(gua, gub)
                    equal_var = lev.pvalue > 0.05
                    
                    t_res = pg.ttest(gua, gub, correction=not equal_var)
                    res.update(t_res.iloc[0].to_dict())
                    res['test'] = f"T-test ({groups[0]} vs {groups[1]})"

            # Mann-Whitney U (2 groups, non-parametric)
            elif test_type == 'mann_whitney':
                groups = df[group_var].astype(str).unique()
                if len(groups) != 2:
                    res['error'] = f"Requires exactly 2 groups, found {len(groups)}"
                else:
                    gua = df[df[group_var].astype(str) == groups[0]][dv].dropna()
                    gub = df[df[group_var].astype(str) == groups[1]][dv].dropna()
                    
                    m_res = pg.mwu(gua, gub)
                    res.update(m_res.iloc[0].to_dict())
                    res['test'] = "Mann-Whitney U"

            # One-way ANOVA (>2 groups)
            elif test_type == 'anova':
                a_res = pg.anova(data=df, dv=dv, between=group_var, detailed=True)
                res.update(a_res.iloc[0].to_dict())
                res['test'] = "One-way ANOVA"
                
            # Kruskal-Wallis (>2 groups, non-parametric)
            elif test_type == 'kruskal':
                k_res = pg.kruskal(data=df, dv=dv, between=group_var)
                res.update(k_res.iloc[0].to_dict())
                res['test'] = "Kruskal-Wallis"

            # Paired T-Test (Before/After) -> Here group_var in UI might be 'Time' or actually we might need 2 cols
            # For simplicity in this structure, we assume user selects 2 dependent variables for paired test 
            # and NO group var (or ignores it). 
            # BUT: Function signature takes group_var. 
            # Let's adjust: For paired tests, we usually compare 2 COLUMNS (Dep Var 1 vs Dep Var 2)
            # So we might need a different UI flow or interpretation.
            # Let's handle Paired separately in the loop if test_type corresponds to it.
            # However, simpler approach: For Paired, dep_vars should contain exactly 2 columns to compare.
            pass 

        except Exception as e:
            res['error'] = str(e)
        
        if test_type not in ['paired_t', 'wilcoxon']:
            results.append(res)

    # Handle Paired Tests (comparing 2 numeric columns directly)
    if test_type in ['paired_t', 'wilcoxon']:
        if len(dep_vars) != 2:
            return [{'error': 'Paired tests require exactly 2 dependent variables selected.'}]
        
        col1 = dep_vars[0]
        col2 = dep_vars[1]
        res = {'variable': f"{col1} vs {col2}"}
        
        try:
            clean_df = df[[col1, col2]].dropna()
            a = clean_df[col1]
            b = clean_df[col2]
            
            if test_type == 'paired_t':
                t_res = pg.ttest(a, b, paired=True)
                res.update(t_res.iloc[0].to_dict())
                res['test'] = "Paired T-test"
            elif test_type == 'wilcoxon':
                w_res = pg.wilcoxon(a, b)
                res.update(w_res.iloc[0].to_dict())
                res['test'] = "Wilcoxon Signed-Rank"
        except Exception as e:
            res['error'] = str(e)
        
        results.append(res)

    return results

def get_variable_types(df):
    """
    Return columns classified by type (numeric, categorical).
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return {
        'numeric': numeric,
        'categorical': categorical
    }

def calculate_correlation(df, method, vars):
    """
    Calculate correlation matrix and return heatmap data.
    method: 'pearson' or 'spearman'
    """
    if not vars or len(vars) < 2:
        return {'error': 'Select at least 2 variables'}
        
    sub_df = clean_numeric_data(df, vars)
    
    if sub_df.shape[1] < 2:
         return {'error': 'Selected variables must be numeric'}

    # Calculate Matrix
    corr_matrix = sub_df.corr(method=method).round(3)
    
    # Calculate P-values (using pingouin uses more resources, simple scipy loop is enough or pandas)
    # Pingouin rcorr is perfect:
    import pingouin as pg
    rcorr = pg.rcorr(sub_df, method=method, stars=False)
    
    # Prepare Heatmap Data (Plotly)
    heatmap = {
        'type': 'heatmap',
        'z': corr_matrix.values.tolist(),
        'x': corr_matrix.columns.tolist(),
        'y': corr_matrix.index.tolist(),
        'colorscale': 'RdBu',
        'zmin': -1,
        'zmax': 1
    }
    
    return {
        'matrix': corr_matrix.reset_index().rename(columns={'index': 'Variable'}).to_dict(orient='records'),
        'heatmap': heatmap,
        'rcorr': rcorr.to_dict(orient='records') # Detailed pairwise with p-values
    }

def calculate_regression(df, method, dv, ivs):
    """
    Calculate Regression (Linear or Logistic).
    method: 'linear' or 'logistic'
    """
    if not dv or not ivs:
        return {'error': 'Missing Dependent or Independent variables'}
        
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Prepare Data
        if method == 'linear':
            data = clean_numeric_data(df, [dv] + ivs)
            X = data[ivs]
            y = data[dv]
        else:
             data = clean_numeric_data(df, [dv] + ivs)
             X = data[ivs]
             y = data[dv]
        
        # Add constant
        X = sm.add_constant(X)
        
        results = {}
        
        if method == 'linear':
            model = sm.OLS(y, X).fit()
            
            # Coefficients Table
            coef_df = pd.DataFrame({
                'Variable': X.columns,
                'Coef (β)': model.params,
                'Std.Error': model.bse,
                't-value': model.tvalues,
                'p-value': model.pvalues
            }).reset_index(drop=True)
            
            # VIF Calculation
            vif_data = []
            if len(ivs) > 1:
                # Exclude constant for VIF usually, but statsmodels includes it if present
                # Let's calc VIF for all columns in X
                for i in range(X.shape[1]):
                    vif_data.append({
                        'Variable': X.columns[i],
                        'VIF': variance_inflation_factor(X.values, i)
                    })
                # Merge VIF into coef_df (simple join logic)
                vif_df = pd.DataFrame(vif_data)
                coef_df = pd.merge(coef_df, vif_df, on='Variable', how='left')
            
            results['summary'] = coef_df.round(4).replace({np.nan: '-'}).to_dict(orient='records')
            results['metrics'] = [
                {'Metric': 'R-squared', 'Value': round(model.rsquared, 3)},
                {'Metric': 'Adj. R-squared', 'Value': round(model.rsquared_adj, 3)},
                {'Metric': 'F-statistic', 'Value': round(model.fvalue, 3)},
                {'Metric': 'Prob (F-statistic)', 'Value': '{:.3e}'.format(model.f_pvalue)}
            ]
            
        elif method == 'logistic':
            # Check if y is binary?
            # Logistic requires discrete y. 
            model = sm.Logit(y, X).fit(disp=0)
            
            coef_df = pd.DataFrame({
                'Variable': X.columns,
                'Coef (β)': model.params,
                'Exp(β) (Odds Ratio)': np.exp(model.params),
                'p-value': model.pvalues
            }).reset_index(drop=True)
            
            results['summary'] = coef_df.round(4).to_dict(orient='records')
            results['metrics'] = [
                {'Metric': 'Pseudo R-squared', 'Value': round(model.prsquared, 3)},
                {'Metric': 'LL-Null', 'Value': round(model.llr_null, 3)},
                 {'Metric': 'LL-Model', 'Value': round(model.llf, 3)}
            ]
            
        return results

        return results

    except Exception as e:
        return {'error': str(e)}

def calculate_efa(df, items, n_factors=None):
    """
    Calculate EFA Stats (KMO, Bartlett, Loadings).
    """
    if not items or len(items) < 3:
        return {'error': 'Select at least 3 items for EFA'}
        
    try:
        from factor_analyzer import FactorAnalyzer
        from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
        
        sub_df = clean_numeric_data(df, items)
        
        # Drop constant columns (Variance = 0) to avoid Singular Matrix
        sub_df = sub_df.loc[:, sub_df.var() > 0]
        if sub_df.shape[1] < 3:
             return {'error': 'Not enough valid items (variance > 0) for EFA.'}

        # KMO & Bartlett
        try:
            kmo_all, kmo_model = calculate_kmo(sub_df)
        except:
             kmo_model = 0 
             
        try:
            chi_square_value, p_value = calculate_bartlett_sphericity(sub_df)
        except:
            p_value = 1.0

        # Fit Factor Analysis
        if not n_factors:
            # Determine factors by Eigenvalues > 1
            fa_temp = FactorAnalyzer(rotation=None)
            fa_temp.fit(sub_df)
            ev, v = fa_temp.get_eigenvalues()
            n_factors = sum(ev > 1)
            if n_factors < 1: n_factors = 1
            
        fa = FactorAnalyzer(n_factors=n_factors, rotation='promax')
        fa.fit(sub_df)
        
        # Loadings (Pattern Matrix)
        loadings = pd.DataFrame(fa.loadings_, index=items, columns=[f'Factor {i+1}' for i in range(n_factors)])
        
        # Variance Explained
        # get_factor_variance returns tuple (SS Loadings, Proportion Var, Cumulative Var)
        var_stats = fa.get_factor_variance()
        variance_df = pd.DataFrame({
            'Factor': [f'Factor {i+1}' for i in range(n_factors)],
            'SS Loadings': var_stats[0],
            '% Variance': var_stats[1] * 100,
            'Cumulative %': var_stats[2] * 100
        })
        
        return {
            'metrics': [
                {'Measure': 'KMO (Kaiser-Meyer-Olkin)', 'Value': round(kmo_model, 3)},
                {'Measure': 'Bartlett Sphericity (p-value)', 'Value': '{:.3e}'.format(p_value)},
                {'Measure': 'Factors Retained', 'Value': int(n_factors)}
            ],
            'loadings': loadings.round(3).reset_index().rename(columns={'index': 'Item'}).to_dict(orient='records'),
            'variance': variance_df.round(3).to_dict(orient='records')
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_sem(df, model_spec):
    """
    Calculate SEM using semopy.
    """
    if not model_spec:
        return {'error': 'Model specification is empty'}
        
    try:
        from semopy import Model
        import semopy
        
        model = Model(model_spec)
        model.fit(df)
        
        # Get Fit Indices
        stats = semopy.calc_stats(model)
        
        # Extract specific fit indices requested
        fit_indices = [
            {'Index': 'CFI', 'Value': round(stats.T['CFI'][0], 3), 'Rule': '> 0.90'},
            {'Index': 'TLI', 'Value': round(stats.T['TLI'][0], 3), 'Rule': '> 0.90'},
            {'Index': 'RMSEA', 'Value': round(stats.T['RMSEA'][0], 3), 'Rule': '< 0.08'},
            {'Index': 'Chi-Square', 'Value': round(stats.T['chi2'][0], 3), 'Rule': 'Low'},
            {'Index': 'P-value', 'Value': '{:.3e}'.format(stats.T['p-value'][0]), 'Rule': '> 0.05'}
        ]
        
        # Estimates
        estimates = model.inspect()
        
        return {
            'fit': fit_indices,
            'estimates': estimates.round(3).to_dict(orient='records')
        }
        
    except Exception as e:
        return {'error': f"SEM Error: {str(e)}"}

    except Exception as e:
        return {'error': str(e)}

def calculate_reliability(df, items):
    """
    Calculate Reliability (Cronbach's Alpha) and Validity (CR, AVE)
    assuming the selected items load on a single factor (common for scale validation).
    """
    if not items or len(items) < 2:
        return {'error': 'Select at least 2 items'}
        
    try:
        import pingouin as pg
        from factor_analyzer import FactorAnalyzer
        
        # USE CLEANER
        sub_df = clean_numeric_data(df, items)
        
        # Cronbach's Alpha
        alpha_res = pg.cronbach_alpha(data=sub_df)
        alpha = alpha_res[0]
        
        # EFA for CR and AVE (Single Factor assumption for simple validity check of a scale)
        fa = FactorAnalyzer(n_factors=1, rotation=None)
        fa.fit(sub_df)
        loadings = fa.loadings_
        
        # Calculate CR and AVE
        # CR = (sum of loadings)^2 / ((sum of loadings)^2 + sum of error variances)
        # AVE = sum of loadings^2 / n_items  .... (approx, usually sum(lambda^2) / n)
        # Actually standard formulas:
        # AVE = (sum(lambda^2)) / n
        # CR = (sum(lambda))^2 / ((sum(lambda))^2 + sum(1-lambda^2))
        
        lam = loadings[:, 0] # loadings vector
        lam_sq = lam ** 2
        error_var = 1 - lam_sq
        
        ave = np.sum(lam_sq) / len(items)
        cr = (np.sum(abs(lam)) ** 2) / ((np.sum(abs(lam)) ** 2) + np.sum(error_var))
        
        return {
            'metrics': [
                {'Measure': "Cronbach's Alpha", 'Value': round(alpha, 3), 'Threshold': '> 0.7'},
                {'Measure': 'Composite Reliability (CR)', 'Value': round(cr, 3), 'Threshold': '> 0.7'},
                {'Measure': 'Average Variance Extracted (AVE)', 'Value': round(ave, 3), 'Threshold': '> 0.5'}
            ],
            'loadings': [
                {'Item': items[i], 'Loading': round(lam[i], 3)} for i in range(len(items))
            ]
        }
        
    except Exception as e:
        return {'error': str(e)}
