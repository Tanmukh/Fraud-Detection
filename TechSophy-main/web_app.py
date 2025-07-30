from flask import Flask, request, render_template_string, redirect, url_for, flash
import os
import pandas as pd
from main import process_claims
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FORM = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Insurance Claims Fraud Detection - Upload</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f8; }
    h1 { color: #2c3e50; }
    form { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); max-width: 400px; }
    input[type=file] { margin-bottom: 10px; }
    input[type=submit] { background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
    input[type=submit]:hover { background-color: #2980b9; }
    ul { color: red; }
  </style>
</head>
<body>
  <h1>Upload Insurance Claims CSV File</h1>
  <form method=post enctype=multipart/form-data action="{{ url_for('upload_file') }}">
    <input type=file name=file>
    <input type=submit value=Upload>
  </form>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <ul>
      {% for message in messages %}
        <li>{{ message }}</li>
      {% endfor %}
      </ul>
    {% endif %}
  {% endwith %}
</body>
</html>
"""

RESULTS_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Insurance Claims Fraud Detection - Results</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f8; }
    h1 { color: #2c3e50; }
    table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 40px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #3498db; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    a { display: inline-block; margin-top: 20px; color: #3498db; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .chart-container { width: 600px; margin-bottom: 40px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
  </style>
</head>
<body>
  <h1>AI CLAIM ANALYZER</h1>

  <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #3498db; background-color: #ecf6fd; width: fit-content; border-radius: 5px;">
    <strong>Flagged Claims:</strong> {{ eval_metrics.num_flagged }} out of {{ eval_metrics.total_claims }} total claims
  </div>

  <table>
    <thead>
      <tr>
        {% for col in columns %}
        <th>{{ col }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for row in data %}
      <tr>
        {% for col in columns %}
        <td>{{ row[col] }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Original Data with Predictions</h2>
  <table>
    <thead>
      <tr>
        {% for col in original_columns %}
        <th>{{ col }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for row in original_data %}
      <tr>
        {% for col in original_columns %}
        <td>{{ row[col] }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Flagged Claims Prioritized for Investigation</h2>
  <table>
    <thead>
      <tr>
        {% for col in flagged_columns %}
        <th>{{ col }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for row in flagged_data %}
      <tr>
        {% for col in flagged_columns %}
        <td>{{ row[col] }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Model Evaluation Metrics</h2>
  <div class="chart-container">
    <canvas id="evalChart"></canvas>
  </div>

  <h2>Fraud Prediction Distribution</h2>
  <div class="chart-container">
    <canvas id="fraudDistChart"></canvas>
  </div>

  <a href="{{ url_for('index') }}">Upload another file</a>

  <script>
    const evalCtx = document.getElementById('evalChart').getContext('2d');
    const evalChart = new Chart(evalCtx, {
      type: 'bar',
      data: {
        labels: ['Precision (0)', 'Recall (0)', 'F1-Score (0)', 'Precision (1)', 'Recall (1)', 'F1-Score (1)', 'Accuracy'],
        datasets: [{
          label: 'Score',
          data: [
            {{ eval_metrics.precision_0 }},
            {{ eval_metrics.recall_0 }},
            {{ eval_metrics.f1_0 }},
            {{ eval_metrics.precision_1 }},
            {{ eval_metrics.recall_1 }},
            {{ eval_metrics.f1_1 }},
            {{ eval_metrics.accuracy }}
          ],
          backgroundColor: [
            '#3498db', '#2980b9', '#1abc9c',
            '#e74c3c', '#c0392b', '#f39c12',
            '#9b59b6'
          ]
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
            max: 1
          }
        }
      }
    });

    // Fraud prediction distribution pie chart
    const fraudDistCtx = document.getElementById('fraudDistChart').getContext('2d');
    const fraudCounts = {
      flagged: {{ original_data | selectattr("fraud_predicted", "equalto", 1) | list | length }},
      total: {{ original_data | length }}
    };
    const fraudDistChart = new Chart(fraudDistCtx, {
      type: 'pie',
      data: {
        labels: ['Flagged Fraud', 'Not Flagged'],
        datasets: [{
          data: [fraudCounts.flagged, fraudCounts.total - fraudCounts.flagged],
          backgroundColor: ['#e74c3c', '#2ecc71']
        }]
      },
      options: {
        responsive: true
      }
    });
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(UPLOAD_FORM)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            data, prioritized, report, num_flagged = process_claims(filepath)
            original_data = data.to_dict(orient='records')
            original_columns = data.columns.tolist()
            flagged_data = prioritized.to_dict(orient='records')
            flagged_columns = prioritized.columns.tolist()
            eval_metrics = {
                'precision_0': round(report['0']['precision'], 2),
                'recall_0': round(report['0']['recall'], 2),
                'f1_0': round(report['0']['f1-score'], 2),
                'support_0': report['0']['support'],
                'precision_1': round(report['1']['precision'], 2),
                'recall_1': round(report['1']['recall'], 2),
                'f1_1': round(report['1']['f1-score'], 2),
                'support_1': report['1']['support'],
                'accuracy': round(report['accuracy'], 2),
                'num_flagged': num_flagged,
                'total_claims': len(data)
            }
            return render_template_string(RESULTS_PAGE, original_data=original_data, original_columns=original_columns, flagged_data=flagged_data, flagged_columns=flagged_columns, eval_metrics=eval_metrics)
        except Exception as e:
            flash(f'Error processing file: {e}')
            return redirect(url_for('index'))
    else:
        flash('Allowed file types are csv')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
