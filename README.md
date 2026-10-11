Tool calls: Running in CPU

intercations will be orchestrated in CPU

agent availability is just like high available of REST API, Orachestrain in Server which coordianted with model running in GPU

Why Multiagent Systems:(MAS)
    Context Management   (Parametric, Non-Parametric knowledge)
    Distributed Development  (moving to micro service agents : as demands it will scale )
    Parallelization
    Scalable
    Strong and Reliable


When to Build MSA:
    -Too many Tool calls : When single agnet has too many tools and struggles to make effective decision on which onces to use 
    -Specialized Knowledge: 
    -Sequesntial Contraints 

Specialized Knowledge?: Parametric or non-parametric that can serve to LLM and make LLM to think , reason and act

Core Principle (MSA): Context Engineering - to decide what information each agent sees. The qualify of system depends on this.

Workflows direct LLM
Agent directs its own Actions/Tasks to achive a goal with a environmental feedback

Popular Frameworks :(MSA)
    JADE
    Autogen
    Langgraph
    Langchain
    Google ADK
    Llamaindex
    CrewAi


Langgraph :
    Built in persistance
    HMI
    Debugging & observability 
    Streaming-first Architecture
    Flexible Agent Patterns

    Key Concepts :(DAG)
        State
        Nodes
        Edges




#Agent  workflow (start , add node,checkpointer,  add  edge , stop )


sub Agents 
    Segregation complexity  (start, add node, checkpointer, add edge, stop)
   from perspective of parnet Agent(A) calling sub agent (B) is a tool calling agent having query and constraints 
   Agent B - based on tool calls or may be RAG synthesize the response and return to Agent A
   what ever B provides still as tool call result to Agent A, Agent A will have to decide how to use the result from Agent B and how to integrate it into its own workflow.
   
   When to use :
    




    