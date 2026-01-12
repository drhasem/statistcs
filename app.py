import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for the current dataset (in memory for simplicity for this user session)
# In a production multi-user env, this would be session-based or database-backed.
CURRENT_DATA = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            if file.filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                return jsonify({'error': 'Invalid file type'}), 400
            
            CURRENT_DATA = df
            
            # Convert NaN to None for JSON compatibility
            preview = df.head(10).replace({np.nan: None}).to_dict(orient='records')
            columns = list(df.columns)
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': file.filename,
                'rows': len(df),
                'columns': columns,
                'preview': preview
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/descriptive', methods=['POST'])
def descriptive_stats():
    global CURRENT_DATA
    if CURRENT_DATA is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        selected_vars = data.get('variables', [])
        
        # Use the dedicated analysis module
        from analysis import get_descriptive_stats
        results = get_descriptive_stats(CURRENT_DATA, selected_vars if selected_vars else None)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/groups', methods=['POST'])
def group_tests():
    global CURRENT_DATA
    if CURRENT_DATA is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        test_type = data.get('test_type')
        group_var = data.get('group_var')
        dep_vars = data.get('dep_vars', [])
        
        if not test_type or not dep_vars:
            return jsonify({'error': 'Missing parameters'}), 400
            
        from analysis import calculate_group_differences
        results = calculate_group_differences(CURRENT_DATA, test_type, group_var, dep_vars)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/variables', methods=['GET'])
def get_variables():
    global CURRENT_DATA
    if CURRENT_DATA is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    from analysis import get_variable_types
    return jsonify(get_variable_types(CURRENT_DATA))

@app.route('/api/correlation', methods=['POST'])
def correlation():
    global CURRENT_DATA
    if CURRENT_DATA is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        method = data.get('method', 'pearson')
        vars = data.get('variables', [])
        
        from analysis import calculate_correlation
        results = calculate_correlation(CURRENT_DATA, method, vars)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/regression', methods=['POST'])
def regression():
    global CURRENT_DATA
    if CURRENT_DATA is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        method = data.get('method')
        dv = data.get('dv')
        ivs = data.get('ivs', [])
        
        from analysis import calculate_regression
        results = calculate_regression(CURRENT_DATA, method, dv, ivs)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reliability', methods=['POST'])
def reliability():
    global CURRENT_DATA
    if CURRENT_DATA is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        items = data.get('items', [])
        
        from analysis import calculate_reliability
        results = calculate_reliability(CURRENT_DATA, items)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/efa', methods=['POST'])
def efa():
    global CURRENT_DATA
    if CURRENT_DATA is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        items = data.get('items', [])
        n_factors = data.get('n_factors') # optional
        if n_factors: n_factors = int(n_factors)
        
        from analysis import calculate_efa
        results = calculate_efa(CURRENT_DATA, items, n_factors)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sem', methods=['POST'])
def sem():
    global CURRENT_DATA
    if CURRENT_DATA is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        model = data.get('model')
        
        from analysis import calculate_sem
        results = calculate_sem(CURRENT_DATA, model)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
