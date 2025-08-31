from typing import TypedDict
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

load_dotenv()

# In-memory storage for todos
todos = []

# Define Todo type
class Todo(TypedDict):
    id: int
    task: str
    completed: bool

# Tool definitions
def add_todo(task: str) -> str:
    """Add a new task to the todo list.
    
    Args:
        task: The task to add to the todo list
        
    Returns:
        Confirmation message
    """
    global todos
    todo_id = len(todos) + 1
    new_todo = {"id": todo_id, "task": task, "completed": False}
    todos.append(new_todo)
    return f"Added todo #{todo_id}: {task}"

def list_todos() -> list[Todo]:
    """List all todos in the todo list.
    
    Returns:
        Formatted list of todos
    """
    global todos
    if not todos:
        return []
    
    return todos
    
    result = "Here's your todo list:\n"
    for todo in todos:
        status = "✅" if todo["completed"] else "⬜️"
        result += f"{status} {todo['id']}. {todo['task']}\n"
    return result

def complete_todo(todo_id: int) -> str:
    """Mark a todo as completed.
    
    Args:
        todo_id: The ID of the todo to mark as complete
        
    Returns:
        Confirmation message
    """
    global todos
    for todo in todos:
        if todo["id"] == todo_id:
            todo["completed"] = True
            return f"Marked todo #{todo_id} as completed."
    return f"Todo with ID {todo_id} not found."

def delete_todo(todo_id: int) -> str:
    """Delete a todo from the list.
    
    Args:
        todo_id: The ID of the todo to delete
        
    Returns:
        Confirmation message
    """
    global todos
    
    # First find the todo item
    todo_to_delete = None
    for todo in todos:
        if todo["id"] == todo_id:
            todo_to_delete = todo
            break
    
    if not todo_to_delete:
        return f"Todo with ID {todo_id} not found."
    
    # Ask for confirmation before deletion
    confirmation = interrupt(f"Are you sure you want to delete todo #{todo_id}: '{todo_to_delete['task']}'? (yes/no)")
    
    # Process based on confirmation - handle both string and list responses
    # Convert confirmation to string if it's a list or other type
    confirmation_str = str(confirmation)
    if isinstance(confirmation, list) and len(confirmation) > 0:
        confirmation_str = str(confirmation[0])
    
    if confirmation_str.lower() in ["yes", "y", "true"]:
        for i, todo in enumerate(todos):
            if todo["id"] == todo_id:
                del todos[i]
                return f"Deleted todo #{todo_id}."
    else:
        return f"Deletion of todo #{todo_id} cancelled."

    
# Create the ReAct agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4.1-nano"),
    tools=[add_todo, list_todos, complete_todo, delete_todo],
    prompt="""You are a helpful todo list manager with vision capabilities.
      You can help users manage their tasks with the following tools:
      - add_todo: Add a new task to the list
      - list_todos: Show all tasks
      - complete_todo: Mark a task as done
      - delete_todo: Remove a task from the list

      You can also see and understand visual content from the user's camera or screen sharing.
      When users share their screen or camera, you can:
      - Describe what you see
      - Help with tasks related to the visual content
      - Provide guidance based on what's displayed
      - Answer questions about documents, images, or applications shown

      Always use the appropriate tool when needed. Be concise but friendly in your responses.
      If you receive visual input, acknowledge it and provide helpful insights.
      """,
    name="todo_agent"
  )