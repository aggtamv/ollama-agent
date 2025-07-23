import os
import sys
import json
import pyfiglet
from rich.text import Text
from rich.panel import Panel
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from master_agent.agent import get_available_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

def main():
    agent = None
    try:
        run = sys.argv[1]
        user_input = run.strip().lower()
        ascii_banner = pyfiglet.figlet_format(f"{user_input} agent")
        print(ascii_banner)

        agent = get_available_agent(user_input)
    except IndexError as e:
        pass

    console = Console()

    if not agent:
        console.print("\n[bold white on black]agent:[/bold white on black] no available agent! Try python main.py gemini|ollama :wave:\n")
        return

    while True:
        user_question = console.input("\n[bold magenta]user:[/bold magenta] ")

        if user_question.lower() == "q":
            console.print("\n[bold white on black]agent:[/bold white on black] have a nice day! :wave:\n")
            break

        inputs = {"messages": [HumanMessage(content=user_question)]}
        config = {"configurable": {"thread_id": "1"}}

        for chunk in agent.stream(inputs, config):
            if "agent" in chunk:
                for message in chunk["agent"]["messages"]:
                    if isinstance(message, AIMessage) and message.content:
                        content = message.content
                        if "create_classifier" in content and "nba_player_stats.csv" in content:
                            from master_agent.tools import create_classifier
                            result = create_classifier("nba_player_stats.csv")
                            time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            console.print(Panel(result, title=f"tool create_classifier {time_now}", style="black on white"))
                        else:
                            time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            console.print(Panel(content, title=f"agent {time_now}", style="black on white"))

            
            if "tools" in chunk:
                for msg in chunk["tools"]["messages"]:
                    if isinstance(msg, ToolMessage):
                        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        content = msg.content
                        name = msg.name

                       
                        console.print(Panel(content, title=f"tool {name} {time_now}", style="black on white"))

        console.print(Text("✓ memory checkpoint created", style="dim italic"))

if __name__ == '__main__':
    main()