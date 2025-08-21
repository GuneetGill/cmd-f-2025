import graphene  # GraphQL library for Python
from .controller.classifier import analyze_prompt_complexity  # Function to classify prompt complexity
from .controller.carbon_footprint import calculate_carbon_footprint # Carbon footprint functions
import json  # For parsing JSON strings

# ----------------------------
# GraphQL Object Types
# ----------------------------

class AnalyzeResult(graphene.ObjectType):
    """
    Defines the structure of the response for the 'analyze' GraphQL query.
    Fields:
    - success: Boolean indicating if the analysis succeeded.
    - query_type: Either "SIMPLE" or "COMPLEX".
    - url: Google search URL if query is SIMPLE, empty otherwise.
    """
    success = graphene.Boolean(default_value=True)
    query_type = graphene.String()
    url = graphene.String()

class CarbonFootprintResult(graphene.ObjectType):
    """
    Defines the structure of the response for the 'calculate_carbon_footprint' GraphQL query.
    Fields:
    - success: Boolean indicating success of the calculation.
    - requests: Number of requests analyzed.
    - carbon_footprint: Calculated carbon footprint value.
    - meaning: Textual interpretation of the carbon footprint.
    """
    success = graphene.Boolean(default_value=True)
    requests = graphene.Int()
    carbon_footprint = graphene.Float()
    meaning = graphene.String()

# ----------------------------
# GraphQL Queries
# ----------------------------

class Query(graphene.ObjectType):
    """
    Defines available GraphQL queries for this API:
    - analyze: Classifies the complexity of a prompt.
    - calculate_carbon_footprint: Calculates carbon footprint for given number of requests.
    """
    analyze = graphene.Field(AnalyzeResult, prompt=graphene.String(required=True))
    calculate_carbon_footprint = graphene.Field(CarbonFootprintResult, requests=graphene.Int(required=True))

    # ----------------------------
    # Resolver for 'analyze' query
    # ----------------------------
    def resolve_analyze(self, info, prompt):
        """
        Handles a request to analyze prompt complexity.
        Steps:
        1. Call the classifier function on the user-provided prompt.
        2. Clean the returned JSON string from formatting characters.
        3. Parse the JSON string into a Python dictionary.
        4. Return an AnalyzeResult object with classification and optional Google URL.
        """
        result_json = analyze_prompt_complexity(prompt)
        # Remove potential newlines, 'json' words, and markdown code fences
        result_json = result_json.replace('\n', '').replace('json','').replace('```', '')
        result = json.loads(result_json)

        # If the response does not contain classification, indicate failure
        if 'classification' not in result:
            return AnalyzeResult(success=False)

        query_type = result['classification']

        # If complex, return classification only
        if query_type == 'COMPLEX':
            return AnalyzeResult(success=True, query_type=query_type)
        # If simple, include Google search URL
        elif result['classification'] == 'SIMPLE':
            url = result['google_search_url']
            return AnalyzeResult(success=True, query_type=query_type, url=url)

        # Default case if something went wrong
        return AnalyzeResult(success=False)

    # ----------------------------
    # Resolver for 'calculate_carbon_footprint' query
    # ----------------------------
    def resolve_calculate_carbon_footprint(self, info, requests):
        """
        Handles a request to calculate carbon footprint.
        Steps:
        1. Compute total carbon footprint for given requests.
        2. Get textual meaning of the footprint.
        3. Return CarbonFootprintResult object with all fields.
        """
        total_carbon_footprint = calculate_carbon_footprint(requests)
        # meaning = get_carbon_footprint_meaning(total_carbon_footprint)
        return CarbonFootprintResult(
            success=True,
            requests=requests,
            carbon_footprint=total_carbon_footprint,
            # meaning=meaning
        )

# ----------------------------
# Create the GraphQL schema
# ----------------------------
schema = graphene.Schema(query=Query)
