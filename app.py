from flask import Flask, render_template, request, redirect, jsonify
from flask_api import status
from flask_cors import CORS
from scripts.lsa import rank_lsa, get_content_from_url

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    url = request.form['article']
    article_content = get_content_from_url(url)
    if article_content is False:
      return render_template('index.html', url_fail=True)
    ranked = rank_lsa(article_content)
    return render_template('index.html', sentences=ranked)
  else:
    return render_template('index.html')
  
@app.route('/api/v1/rank', methods=['GET'])
def api_filter():
    query_parameters = request.args

    url = query_parameters.get('url')
    
    if not url:
      return "Record not found", status.HTTP_400_BAD_REQUEST
    
    article_content = get_content_from_url(url)
    if article_content is False or article_content == "":
      return "Record not found", status.HTTP_400_BAD_REQUEST
    ranked = rank_lsa(article_content)

    return jsonify(ranked)

if __name__ == "__main__":
  app.run(debug=True)