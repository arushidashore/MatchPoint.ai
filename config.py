import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-super-secret-key-change-this'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///matchpoint.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'static'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

class ProductionConfig(Config):
    DEBUG = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'your-jwt-secret-key-change-this'

class DevelopmentConfig(Config):
    DEBUG = True
    JWT_SECRET_KEY = 'dev-secret-key'
