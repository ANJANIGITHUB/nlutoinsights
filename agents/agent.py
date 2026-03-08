import os
import sys
import json
import psycopg
from typing import Dict, Any, Optional

# --- ADK imports ---
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
from google.genai import types as genai_types
from dotenv import load_dotenv
from neo4j import GraphDatabase
import openai
from sqlalchemy import create_engine, text

import litellm
litellm.drop_params = True

load_dotenv()
#MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash-lite") # Any supported model string works.
#MODEL=LiteLlm(model=os.environ.get("OPENAI_API_KEY"))

print(os.getenv("OPENAI_API_KEY"))

MODEL = LiteLlm(
    model="openai/gpt-5-nano",
    temperature=1    
)

# PostgreSQL env vars (must exist in .env)
PGHOST = os.environ["PGHOST"]
PGPORT = int(os.environ.get("PGPORT", 5432))  # default only for port
PGDATABASE = os.environ["PGDATABASE"]
PGUSER = os.environ["PGUSER"]
PGPASSWORD = os.environ["PGPASSWORD"]

POSTGRES_URI = "postgresql://postgres:0307@localhost:5432/aidb"



#Neo4J Connection details
NEO4J_URI=os.environ["NEO4J_URI"]
NEO4J_USERNAME=os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD=os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE=os.environ["NEO4J_DATABASE"]
AURA_INSTANCEID=os.environ["AURA_INSTANCEID"]
AURA_INSTANCENAME=os.environ["AURA_INSTANCENAME"]

AUTH=(NEO4J_USERNAME,NEO4J_PASSWORD)

#create driver for graph database
driver=GraphDatabase.driver(NEO4J_URI,auth=AUTH)

# ==========================================================
# EMBEDDING SECTION
# ==========================================================

def generate_embedding(text_input):
    """
    Generates embedding vector for:
        - Table descriptions
        - Column descriptions
        - User questions

    Returns:
        List[float] (1536 dimensional vector)
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text_input
    )

    return response.data[0].embedding


class SchemaGraph:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def create_schema(self):

        with self.driver.session() as session:

            # =====================================================
            # CREATE SCHEMA NODE
            # =====================================================

            session.run("""
            MERGE (s:Schema {name:'aischema'})
            """)

            # =====================================================
            # CUSTOMERS TABLE
            # =====================================================

            customers_desc = """
            aischema.customers table containing:
            customer_id (primary key),
            first_name,
            last_name,
            email,
            phone,
            city,
            created_at
            """

            customers_embedding = generate_embedding(customers_desc)

            session.run("""
            MERGE (t:Table {name:'customers'})
            SET t.full_name='aischema.customers',
                t.description=$desc,
                t.embedding=$embedding
            """, desc=customers_desc,
                 embedding=customers_embedding)

            session.run("""
            MATCH (s:Schema {name:'aischema'})
            MATCH (t:Table {name:'customers'})
            MERGE (s)-[:HAS_TABLE]->(t)
            """)

            # Customers Columns
            customers_columns = [
                ("customer_id", "integer"),
                ("first_name", "varchar"),
                ("last_name", "varchar"),
                ("email", "varchar"),
                ("phone", "varchar"),
                ("city", "varchar"),
                ("created_at", "timestamp")
            ]

            for col_name, data_type in customers_columns:
                session.run("""
                MERGE (c:Column {full_name:$full_name})
                SET c.name=$name,
                    c.data_type=$type
                """,
                full_name=f"aischema.customers.{col_name}",
                name=col_name,
                type=data_type)

                session.run("""
                MATCH (t:Table {name:'customers'})
                MATCH (c:Column {full_name:$full_name})
                MERGE (t)-[:HAS_COLUMN]->(c)
                """,
                full_name=f"aischema.customers.{col_name}")

            # =====================================================
            # PRODUCTS TABLE
            # =====================================================

            products_desc = """
            aischema.products table containing:
            name,
            description,
            price,
            category_id
            """

            products_embedding = generate_embedding(products_desc)

            session.run("""
            MERGE (t:Table {name:'products'})
            SET t.full_name='aischema.products',
                t.description=$desc,
                t.embedding=$embedding
            """, desc=products_desc,
                 embedding=products_embedding)

            session.run("""
            MATCH (s:Schema {name:'aischema'})
            MATCH (t:Table {name:'products'})
            MERGE (s)-[:HAS_TABLE]->(t)
            """)

            products_columns = [
                ("name", "varchar"),
                ("description", "text"),
                ("price", "numeric"),
                ("category_id", "integer")
            ]

            for col_name, data_type in products_columns:
                session.run("""
                MERGE (c:Column {full_name:$full_name})
                SET c.name=$name,
                    c.data_type=$type
                """,
                full_name=f"aischema.products.{col_name}",
                name=col_name,
                type=data_type)

                session.run("""
                MATCH (t:Table {name:'products'})
                MATCH (c:Column {full_name:$full_name})
                MERGE (t)-[:HAS_COLUMN]->(c)
                """,
                full_name=f"aischema.products.{col_name}")

        print("Production-grade schema graph created successfully.")

    def create_vector_index(self):

        with self.driver.session() as session:
            session.run("""
            CREATE VECTOR INDEX table_embedding_index
            IF NOT EXISTS
            FOR (t:Table)
            ON (t.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """)

        print("Vector index created.")

# ==========================================================
# VECTOR SEARCH (GraphRAG - Production Grade)
# ==========================================================

def retrieve_relevant_schema(question_embedding, top_k=5):
    """
    Performs vector similarity search in Neo4j.

    Searches:
        - Table embeddings
        - Column embeddings (optional if added later)

    Returns:
        {
            "tables": [full_table_names],
            "columns": [full_column_names]
        }
    """

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    with driver.session() as session:

        # ------------------------------------------------------
        # 1️⃣ Search Tables
        # ------------------------------------------------------
        table_result = session.run("""
        CALL db.index.vector.queryNodes(
            'table_embedding_index',
            $top_k,
            $embedding
        )
        YIELD node, score
        RETURN node.full_name AS full_name, score
        ORDER BY score DESC
        """, embedding=question_embedding, top_k=top_k)

        tables = [record["full_name"] for record in table_result]

        # ------------------------------------------------------
        # 2️⃣ (Optional) Search Columns if you create index later
        # ------------------------------------------------------
        columns = []
        try:
            column_result = session.run("""
            CALL db.index.vector.queryNodes(
                'column_embedding_index',
                $top_k,
                $embedding
            )
            YIELD node, score
            RETURN node.full_name AS full_name, score
            ORDER BY score DESC
            """, embedding=question_embedding, top_k=top_k)

            columns = [record["full_name"] for record in column_result]

        except Exception:
            # Column index might not exist yet
            pass

    driver.close()

    return {
        "tables": list(set(tables)),
        "columns": list(set(columns))
    }



# -----------------------------------------------------------------------------
# Tool: Safe, read-only SQL executor for PostgreSQL
# -----------------------------------------------------------------------------

def run_postgres_sql(sql: str) -> dict[str, any]:
    """
    Execute a single, read-only SELECT statement on PostgreSQL and return rows.

    Args:
    sql: The EXACT SQL statement to execute. Must start with SELECT and be a single statement
    (no semicolons in the middle).

    Returns:
    {
    "status": "success" | "error",
    "columns": [col, ...], # present on success
    "rows": [[...], ...], # present on success
    "error_message": "..." # present on error
    }
    """
    try:
        sql_clean = sql.strip()

        # Guardrails: only allow a single SELECT statement
        if not sql_clean.lower().startswith("select"):
            return {"status": "error", "error_message": "Only SELECT statements are allowed."}
        if ";" in sql_clean.rstrip(";"):
            return {"status": "error", "error_message": "Multiple statements are not allowed."}

        conn = psycopg.connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        )
        cur = conn.cursor()
        cur.execute(sql_clean)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description]
        cur.close()
        conn.close()
        return {"status": "success", "columns": columns, "rows": rows}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# -----------------------------------------------------------------------------
# Agent 1: Planner — produce a structured SQL plan from the user question
# The plan is saved to session.state["sql_plan"] via output_key. (SequentialAgent will
# pass shared state to the next agent in the same turn.)
# -----------------------------------------------------------------------------
planner_agent = LlmAgent(
    name="SqlPlanner",
    model=MODEL,
    description="Determines whether to answer normally or generate SQL plan.",
    instruction="""
You are an intelligent assistant.

If the user message is greeting, small talk, or unrelated to database queries:
Return JSON:
{
"skip": true,
"response": "Natural helpful reply to the user"
}

If the message is asking about:

products

price

category

product listing

product filtering

product searching

Return JSON:
{
"skip": false,
"plan": {
"tables": ["aischema.products"],
"join_condition": null,
"filters": "...",
"columns": "...",
"aggregation": "..."
}
}

If the message is asking about:

customers

customer details

customer listing

customer filtering

customer searching

Return JSON:
{
"skip": false,
"plan": {
"tables": ["aischema.customers"],
"join_condition": null,
"filters": "...",
"columns": "...",
"aggregation": "..."
}
}

If the user query requires both product and customer information (for example: which customer purchased which product, customer purchases, products bought by a customer, or customer-product relationships), use BOTH tables and include a JOIN.

Return JSON:
{
"skip": false,
"plan": {
"tables": ["aischema.products","aischema.customers"],
"join_condition": "aischema.products.customer_purchased = aischema.customers.first_name",
"filters": "...",
"columns": "...",
"aggregation": "..."
}
}

Database:
Schema: aischema

Table: aischema.products
Columns:
"id" "integer"
"name" "character varying"
"description" "text"
"price" "numeric"
"category_id" "integer"
"customer_purchased" "character varying"

Table: aischema.customers
Columns:
"customer_id" "integer"
"first_name" "character varying"
"last_name" "character varying"
"email" "character varying"
"phone" "character varying"
"city" "character varying"
"created_at" "timestamp without time zone"

Relationship:
aischema.products.customer_purchased = aischema.customers.first_name

Use this relationship when both tables are required.

Return ONLY valid JSON.""",
    output_key="sql_plan",
)



# -----------------------------------------------------------------------------
# Agent 2: Executor — turn plan into SQL, run Postgres tool, return results
# The final assistant message is stored in session.state["query_result"].
# -----------------------------------------------------------------------------
executor_agent = LlmAgent(
    name="SqlExecutor",
    model=MODEL,
    description="Executes SQL or returns general response.",
    instruction="""
You receive sql_plan from shared state.

If sql_plan.skip == true:
Return sql_plan.response directly.

If sql_plan.skip == false:
1. Convert the plan into exactly ONE SELECT statement.
2. Use aischema.products and/or
3. Use aischema.Customers
4. Call the SQL tool.
5. Format results clearly for the user.

Only generate one SELECT statement.
""",
    tools=[run_postgres_sql],
)



# ==========================================================
# POSTGRES EXECUTION
# ==========================================================

# def execute_sql(query):
#     """
#     Executes SQL in PostgreSQL safely.

#     Returns:
#         List of rows
#     """

#     engine = create_engine(POSTGRES_URI)

#     with engine.connect() as conn:
#         result = conn.execute(text(query))
#         return result.fetchall()


# # ==========================================================
# # FULL PIPELINE
# # ==========================================================

# def process_question(question):
#     """
#     Complete GraphRAG pipeline:

#     1. Planner Agent creates plan
#     2. Embed user question
#     3. Retrieve relevant tables
#     4. SQL Agent generates query
#     5. Execute query
#     """

#     print("\nSTEP 1: Planning")
#     plan = planner_agent.run(question)
#     print(plan)

#     print("\nSTEP 2: Embedding Question")
#     question_embedding = generate_embedding(question)

#     print("\nSTEP 3: Vector Search")
#     relevant_tables = retrieve_relevant_schema(question_embedding)
#     print("Relevant Tables:", relevant_tables)

#     print("\nSTEP 4: SQL Generation")
#     sql_query = executor_agent.run(f"""
#     User Question: {question}
#     Plan: {plan}
#     Relevant Tables: {relevant_tables}
#     """)
#     print(sql_query)

#     print("\nSTEP 5: Executing SQL")
#     results = execute_sql(sql_query)

#     return results


# ==========================================================
# INITIALIZATION
# ==========================================================

def initialize_system():
    """
    Run once:
        - Create schema graph
        - Create vector index
    """

    graph = SchemaGraph()
    graph.create_schema()
    graph.create_vector_index()

initialize_system()

root_agent = SequentialAgent(
name="SqlPipeline",
sub_agents=[planner_agent, executor_agent],
)

root_agent


# ==========================================================
# RUN
# ==========================================================

# if __name__ == "__main__":

#     # Run once (comment after first execution)
#     initialize_system()

#     user_question = "Show all customers"

#     output = process_question(user_question)

#     print("\nFINAL RESULT:")
#     print(output)
