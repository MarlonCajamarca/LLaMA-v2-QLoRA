import os
import pandas as pd
from neo4j import GraphDatabase, Transaction

from dotenv import load_dotenv

class Neo:
    def __init__(self):
        self.create_neo4j_driver()
        self.graph_schema = dict()

    def create_neo4j_driver(self):
        load_dotenv()
        target_uri = os.getenv("NEO4J_TARGET_URI")
        target_user = os.getenv("NEO4J_TARGET_USER")
        target_password = os.getenv("NEO4J_TARGET_PASSWORD")
        target_database = os.getenv("NEO4J_TARGET_DATABASE")
        self.neo4j_driver = GraphDatabase.driver(
            target_uri, 
            auth=(target_user, 
                  target_password),
            database=target_database)
        try:
            self.neo4j_driver.verify_connectivity()
        except Exception as e:
            print(e)
            raise Exception("Unable to connect to Neo4j database")

    def get_graph_schema(self):
        with self.neo4j_driver.session() as session:
            # Get all valid (node)-[relationship]-(node) ontology paths for given graph database
            node_relationships_list = session.execute_read(self._get_node_relationships_schema)
            # Store the node_relationships_list in the graph_schema dictionary
            self.graph_schema["node_relationships"] = node_relationships_list
            # Get all valid node labels for given graph database
            node_properties_dict = session.execute_read(self._get_node_properties_schema)
            # Store the node_properties_dict in the graph_schema dictionary
            self.graph_schema["node_properties"] = node_properties_dict
            # Get all valid relationship properties for given graph database
            relationship_properties_dict = session.execute_read(self._get_relationship_properties_schema)
            # Store the relationship_properties_dict in the graph_schema dictionary
            self.graph_schema["relationship_properties"] = relationship_properties_dict
        return self.graph_schema

    def _get_relationship_properties_schema(self, tx: Transaction):
        relationship_properties_query = """
        MATCH (n)-[rels]-(m) 
        WITH DISTINCT type(rels) as relationships, apoc.coll.sort(keys(rels)) as properties 
        RETURN relationships AS relationship_label, apoc.coll.toSet(apoc.coll.flatten(collect(properties))) AS relationship_properties
        """
        # Execute the relationship_properties_query and store the results in a dataframe
        query_result = tx.run(relationship_properties_query)
        relationship_properties_df = query_result.to_df()
        # Create a key value dictionary with relationship_label as key and relationship_properties as value
        relationship_properties_dict = dict(zip(relationship_properties_df["relationship_label"], relationship_properties_df["relationship_properties"]))
        # assert that type of keys is string and type of values is list
        assert all(isinstance(key, str) for key in relationship_properties_dict.keys())
        assert all(isinstance(value, list) for value in relationship_properties_dict.values())
        return relationship_properties_dict

    def _get_node_properties_schema(self, tx: Transaction):
        node_properties_query = """
        //Match all nodes from the graph database
        MATCH (n)
        //Return the node labels and corresponding properties for each node. Properties are sorted alphabetically 
        WITH DISTINCT LABELS(n) as nodes, apoc.coll.sort(keys(n)) as properties
        //Return the node label and a list of all node properties
        RETURN nodes[0] AS node_label, apoc.text.join(apoc.coll.toSet(apoc.coll.flatten(collect(properties))),', ') AS node_properties
        """
        # Execute the node_properties_query and store the results in a dataframe
        query_result = tx.run(node_properties_query)
        node_properties_df = query_result.to_df()
        # Cast the node_properties column to a list of strings
        node_properties_df["node_properties"] = node_properties_df["node_properties"].apply(lambda x: x.split(", "))
        # Create a key value dictionary with node_label as key and node_properties as value
        node_properties_dict = dict(zip(node_properties_df["node_label"], node_properties_df["node_properties"]))
        # assert that type of keys is string and type of values is list
        assert all(isinstance(key, str) for key in node_properties_dict.keys())
        assert all(isinstance(value, list) for value in node_properties_dict.values())
        return node_properties_dict

    def _get_node_relationships_schema(self, tx: Transaction):
        node_relationships_query = """
        //Match all valid paths from the graph database
        MATCH (n)-[rels]->(m) 
        //Return the relationship labels and corresponding origin and destination node labels for each path
        WITH DISTINCT TYPE(rels) AS relations, LABELS(n) AS origin, LABELS(m) AS destination
        //Return the origin and destination node labels and the relationship label for each valid path
        RETURN DISTINCT apoc.text.join(["(:", origin[0], ")-[:", relations,"]->(:", destination[0], ")"], "") AS node_relationships
        """
        # Execute the node_relationships_query and store the results in a dataframe
        query_result = tx.run(node_relationships_query)
        node_relationships_df = query_result.to_df()
        # take dataframe and convert to list of strings
        node_relationships_list = node_relationships_df["node_relationships"].tolist()
        # assert that type is list
        assert isinstance(node_relationships_list, list)
        return node_relationships_list

    def get_neo4j_driver(self):
        return self.neo4j_driver

class SeedPromptAugmenter(object):
    def __init__(self):
        self.neo4j = Neo()
        self.graph_schema = self.neo4j.get_graph_schema()
        self.system_prompt = str()
        self.system_instruction_prompt = str()
        self.graph_schema_prompt = str()
        self.system_user_request_prompt = str()
        self.user_query = str()

    def augment(self, user_query):
        # Set the system prompt.
        self.set_system_instruction_prompt()
        self.set_graph_schema_prompt()
        self.set_system_user_request_prompt()
        self.system_prompt = f"""{self.system_instruction_prompt}\n\n{self.graph_schema_prompt}\n{self.system_user_request_prompt}"""
        # Set the user query prompt.
        self.user_query = user_query
    
    def set_system_user_request_prompt(self):
        self.system_user_request_prompt = """With this information in hand, and considering your expert knowledge of Cypher, generate the appropriate Cypher queries based on the following English instruction:"""

    def set_system_instruction_prompt(self):
        role_prompt = """You are an expert Cypher developer."""
        task_prompt = """Your job is to convert English instructions into valid Cypher queries that are in sync with the given graph database's structure."""
        result_format_prompt = """Your result must be a Cypher file delimited by ```cypher delimiter and must not contain any textual description of the query nor any commented block within the generated query."""
        instruction_prompt = """To achieve this, make use of the following schema details and example queries to guide your understanding:"""
        self.system_instruction_prompt = f"""{role_prompt}.{task_prompt}.{result_format_prompt}.{instruction_prompt}"""

    def set_graph_schema_prompt(self):
        node_relationships_prompt = f"""Python list of valid (node)-[relationships]-(node) paths/ontologies for the given graph database.You must take special account to the relationship direction for each (node)-[relationship]-(node) path in the given schema. Posible relationship directions, delimited by backticks, are:`()-[]-()`, `()->[]-()`, `()-[]->()`, `()-[]<-()`, `()<-[]-()`,  `()->[]<-()`\nnode_relationships_list={self.graph_schema["node_relationships"]}\n"""
        node_properties_prompt = f"""Python dictionary of valid node labels as keys and corresponding node properties as values,for the given graph database.\nnode_properties_dictionary={self.graph_schema["node_properties"]}\n"""
        relationship_properties_prompt = f"""Python dictionary of valid relationship labels as keys and corresponding relationship properties as values, for the given graph database. If a given relationship does not have porperties, the corresponding value will be an empty list.\nrelationship_properties_dictionary={self.graph_schema["relationship_properties"]}\n"""
        self.graph_schema_prompt = f"""Neo4J Graph Database Schema\nThe graph schema for the given Neo4J graph database contains all the nodes, relationships and their corresponding properties. The most important part of the schema are the  Node-Relationships, which are all the valid paths or ontologies within the Neo4J graph database that you must follow to obtain valid Cypher queries. This is the Neo4J graph database schema:\n\nNode-Relationships: {node_relationships_prompt}\nNode Properties: {node_properties_prompt}\nRelationship Properties: {relationship_properties_prompt}"""
    

if __name__ == "__main__":
    user_query = "How many products level 0 are there?"
    seed_augmenter = SeedPromptAugmenter()
    seed_augmenter.augment(user_query)
    print("*"*100)
    print(seed_augmenter.system_prompt)
    print("*"*100)
    print(seed_augmenter.user_query)
    print("*"*100)