# import dependencies


#Agent  workflow (start , add node,checkpointer,  add  edge , stop )


# sub Agents 
 #  Segregation complexity  (start, add node, checkpointer, add edge, stop)
 #  from perspective of parnet Agent(A) calling sub agent (B) is a tool calling agent having query and constraints 
 #  Agent B - based on tool calls or may be RAG synthesize the response and return to Agent A
 #  what ever B provides still as tool call result to Agent A, Agent A will have to decide how to use the result from Agent B and how to integrate it into its own workflow.
 #  At any point Agent A and Agent B has its own persistence and checkpointer to store its own state and progress.
 #  