#!/usr/bin/env python3
"""
Archaeological Pipeline Web Interface

Flask-based web application for managing the AI-powered archaeological
detection pipeline with real-time monitoring and parameter editing.

Authors: Archaeological AI Team
License: MIT
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import yaml
import subprocess
import threading
import sys
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'archaeological_discovery_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

current_process = None

def load_parameters():
    """Load parameters from YAML configuration file."""
    with open("config/parameters.yaml", 'r') as f:
        return yaml.safe_load(f)

def save_parameters(params):
    """Save parameters to YAML configuration file."""
    with open("config/parameters.yaml", 'w') as f:
        yaml.dump(params, f, default_flow_style=False, indent=2)

def stream_process_output(process, script_name):
    """Stream process output to WebSocket clients."""
    for line in iter(process.stdout.readline, ''):
        if line:
            output_line = line.strip()
            socketio.emit('script_output', {
                'script': script_name,
                'output': output_line,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
    
    return_code = process.wait()
    socketio.emit('script_complete', {
        'script': script_name,
        'success': return_code == 0,
        'return_code': return_code,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })

@app.route('/')
def index():
    """Main interface page."""
    parameters = load_parameters()
    return render_template('index.html', parameters=parameters)

@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """API endpoint to get current parameters."""
    return jsonify(load_parameters())

@app.route('/api/parameters', methods=['POST'])
def update_parameters():
    """API endpoint to update parameters."""
    save_parameters(request.json)
    return jsonify({'success': True, 'message': 'Parameters saved successfully'})

@app.route('/api/run_script', methods=['POST'])
def run_script():
    """API endpoint to execute pipeline scripts."""
    global current_process
    
    script_type = request.json.get('script_type')
    
    script_commands = {
        'setup': [sys.executable, 'setup_pipeline.py'],
        'pipeline': [sys.executable, 'run_pipeline.py', '--full'],
        'checkpoint': [sys.executable, 'run_checkpoint.py'],
        'visualization': [sys.executable, 'result_visualization.py']
    }
    
    # Set UTF-8 encoding environment to handle Unicode characters
    import os
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'  # Ensure immediate output flushing
    
    current_process = subprocess.Popen(
        script_commands[script_type],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',
        bufsize=1,
        env=env
    )
    
    output_thread = threading.Thread(
        target=stream_process_output,
        args=(current_process, script_type)
    )
    output_thread.daemon = True
    output_thread.start()
    
    return jsonify({
        'success': True, 
        'message': f'{script_type.title()} script started',
        'pid': current_process.pid
    })

@app.route('/api/stop_script', methods=['POST'])
def stop_script():
    """API endpoint to stop the currently running script."""
    global current_process
    
    if current_process and current_process.poll() is None:
        current_process.terminate()
        current_process.wait(timeout=5)
        return jsonify({'success': True, 'message': 'Script stopped'})
    return jsonify({'success': False, 'error': 'No script is currently running'}), 400

@app.route('/api/status')
def get_status():
    """API endpoint to get current pipeline status."""
    return jsonify({
        'script_running': current_process is not None and current_process.poll() is None,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    emit('status', {'message': 'Connected to Archaeological Pipeline Interface'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)