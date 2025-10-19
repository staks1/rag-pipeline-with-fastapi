# rag-pipeline-with-fastapi
A project designed to create a rag pipeline with vector database and fastapi backend


## Main RAG chunking Pipeline

1. Split a document of a topic in different N chunks, sharing the same metadata "source"
    - The split is based on a context window max size we set, allowing each document to be split along N chunks, each one not exceeding the max window context size.
2. Embed the query using the embedder model selected
3. Perform similarity search in the vector store selecting the top-k scoring groups of documents using a certain top-k value and a certain threshold for scores to consider (group query).
    - The assumption is that for a certain topic, all collected chunks share the same metadata "source" key.
    - So we select all possible groups and take the top-1 scoring for each group.
    - Compare the differences between all top 1 group scores.
    - If the difference between top 1 and top 2 is important then we have a clear document winner
    - If the resulting group is only 1 then again we have a clear document winner .
    - In case of clear document winner we take the group_id (document source) returned and, 
    - Again perform query search but now we search for all chunks of this group without any threshold (potentially all are useful as context).
    - Now that we have all chunks of the winner document, we sort them and consider as many as possible such that the total context does not exceed the context length window we set .
    - We construct the new context (received from RAG) this way and send the query + new contect to the llm to get the result.
    - In case there is no document winner (meaning no vector returned because of no score exceeding the score threshold, or very small differences between top 1 groups' scores) we return "Do not know" anwser so that we do not return invalid answers to the user. In that case the user needs to modify his question to be more concise and/or rephrase it.
    
