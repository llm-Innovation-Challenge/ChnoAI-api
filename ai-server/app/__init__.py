from flask import Flask
import logging

# 블루프린트 등록
from app.routes import bp as main_routes_bp
from app.categorize_questions import categorize_questions_bp
from app.summarize_answers import summarize_answers_bp
from app.draft_implementation_blog import draft_implementation_blog_bp
from app.draft_debugging_blog import draft_debugging_blog_bp
from app.draft_explanation_blog import draft_explanation_blog_bp
from app.review_and_finalize_blog import review_and_finalize_blog_bp
from app.publish_to_notion import publish_to_notion_bp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app():
    app = Flask(__name__)
    
    # 블루프린트 등록
    app.register_blueprint(main_routes_bp)
    app.register_blueprint(categorize_questions_bp, url_prefix='/')
    app.register_blueprint(summarize_answers_bp, url_prefix='/')
    app.register_blueprint(draft_implementation_blog_bp, url_prefix='/')
    app.register_blueprint(draft_debugging_blog_bp, url_prefix='/')
    app.register_blueprint(draft_explanation_blog_bp, url_prefix='/')
    app.register_blueprint(review_and_finalize_blog_bp, url_prefix='/')
    app.register_blueprint(publish_to_notion_bp, url_prefix='/')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
    