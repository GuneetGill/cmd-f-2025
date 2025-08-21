from flask import Flask
from flask_cors import CORS
from graphene import Schema
from flask_graphql import GraphQLView
from .schema import schema

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Sample route
    @app.route('/')
    def hello():
        return {'message': 'API is running'}
    
    app.add_url_rule(
        '/graphql', 
        view_func=GraphQLView.as_view('graphql', schema=schema, graphiql=True)
    )
    return app

# Create the application instance
app = create_app()

