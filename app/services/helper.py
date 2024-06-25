import os
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


def data_genie(query):
    print("Data Query ==> ", query)
    return str("data genie")


def insights_genie(query):
    print("insight Query ==> ", query)
    return str("insight genie0")


def campaign_genie(query):
    print("campaign Query ==> ",query)
    return str("campaign genie")


client_azure = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),  # Your API key for the assistant api model
    api_version=os.getenv("AZURE_API_KEY_VERSION"),  # API version  (i.e. 2024-02-15-preview)
    azure_endpoint=os.getenv("AZURE_ENDPOINT"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "data_genie",
            "description": "any data related query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "user question on data"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "insights_genie",
            "description": "gives you insights on user question",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "user question"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "campaign_genie",
            "description": "gives you trends, advertisement campaign and marketing strategy",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "user query"
                    }
                },
                "required": ["query"]
            }
        }
    },
]


def create_assistant():
    assistant = client_azure.beta.assistants.create(
        name="Layer Assistant",
        model=os.getenv("AZURE_MODEL_NAME"),
        instructions="You are a personal AI Assistant",
        tools=tools)

    return assistant


# a thread that runs message and openAI query
def create_message_and_run(assistant_id, query, thread_id):

    message = client_azure.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=query
    )
    run = client_azure.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions='You are a multi-layered assistant. You take the user\'s query, find the right function to handle'
                     'it, and if there\'s no match, you use the default assistant.',
    )
    return run


def retrieve_run(run, thread_id):
    while True:
        # Wait for 5 seconds
        time.sleep(5)

        # Retrieve the run status
        run_status = client_azure.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        print("run_status", run_status)
        print("run_status.status", run_status.status)
        if run_status.status == 'completed':
            break
        elif run_status.status == 'requires_action':
            print("Function Calling")
            required_actions = run_status.required_action.submit_tool_outputs.model_dump()
            print("required_actions", required_actions)
            tool_outputs = []
            import json
            for action in required_actions["tool_calls"]:
                func_name = action['function']['name']
                arguments = json.loads(action['function']['arguments'])
                output = globals()[func_name](**arguments)  # Assuming functions are in the global scope
                print("output", output)
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output
                })

            print("Submitting outputs back to the Assistant...")
            client_azure.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs)


def send_message_and_retrieve_response(query, thread_id, assistant_id):
    run = create_message_and_run(assistant_id, query, thread_id)
    retrieve_run(run, thread_id)
    all_messages = client_azure.beta.threads.messages.list(thread_id=thread_id)
    print("all_messages", all_messages)
    return all_messages.data[0].content[0].text.value


def create_thread():
    thread = client_azure.beta.threads.create()
    return thread.id


def layer(query):
    messages = [
        {"role": "system", "content": 'You are a multi-layered assistant. You take the user\'s query, find the right function to handle'
                                      'it, and if there\'s no match, you use the default assistant.'},
        {"role": "user", "content": query},
    ]
    response = client_azure.chat.completions.create(
        model=os.getenv("AZURE_MODEL_NAME"),
        messages=messages,
        temperature=0,
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message
    messages.append(response_message)

    print(response_message)
    tool_calls = response_message.tool_calls
    if tool_calls:
        # If true the model will return the name of the tool / function to call and the argument(s)
        tool_call_id = tool_calls[0].id
        tool_function_name = tool_calls[0].function.name
        tool_query_string = eval(tool_calls[0].function.arguments)['query']

        # Step 3: Call the function and retrieve results. Append the results to the messages list.
        if tool_function_name == 'data_genie':
            results = data_genie(tool_query_string)

            messages.append({
                "role":"tool",
                "tool_call_id":tool_call_id,
                "name": tool_function_name,
                "content":results
            })

            # Step 4: Invoke the chat completions API with the function response appended to the messages list
            # Note that messages with role 'tool' must be a response to a preceding message with 'tool_calls'
            model_response_with_function_call = client_azure.chat.completions.create(
                model=os.getenv("AZURE_MODEL_NAME"),
                messages=messages,
            )  # get a new response from the model where it can see the function response
            return model_response_with_function_call.choices[0].message.content
        else:
            print(f"Error: function {tool_function_name} does not exist")
    else:
        # Model did not identify a function to call, result can be returned to the user
        print(response_message.content)
