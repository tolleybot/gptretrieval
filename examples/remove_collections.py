from pymilvus import Milvus, connections

# Step 1: Establish a connection to your Milvus instance
connections.connect(host='localhost', port='19530', alias='default')

# Step 2: Retrieve the list of all collections
milvus = Milvus()
collections = milvus.list_collections()

# Step 3: Iterate through the list of collections and delete each one
for collection_name in collections:
    milvus.drop_collection(collection_name)

# Close the connection
connections.remove_connection(alias='default')

