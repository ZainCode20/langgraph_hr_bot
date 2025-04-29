import os
import streamlit as st
from typing import Dict, Any
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END # Keep StateGraph and END import for graph definition

# Questions list
questions = [
    "What is your name?",
    "What is your professional background?",
    "What are your technical skills?",
    "What are your strengths?",
    "What are your weaknesses?",
    "Describe a challenging project you completed.",
    "What are your career goals?",
    "How do you handle stress?",
    "How do you work in a team?",
    "Why should we hire you?"
]

# Setup LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # Using a common model name
    temperature=0.5,
    api_key=st.secrets["auth_token"]
)

# --- Streamlit Session State Initialization ---
# This runs only once per user session
if 'graph_state' not in st.session_state: # Store the core interview state (answers)
    st.session_state.graph_state = {"answers": []}
if 'chat_history' not in st.session_state: # History for displaying conversation
    st.session_state.chat_history = []
if 'interview_complete' not in st.session_state: # Flag to indicate end of interview
    st.session_state.interview_complete = False
if 'report_generated' not in st.session_state: # Flag to prevent report regeneration
    st.session_state.report_generated = False
if 'interview_started' not in st.session_state: # New flag to track if interview has started
    st.session_state.interview_started = False




# Node 1: Ask the next question (Logic)
def ask_question_logic(state: Dict[str, Any]) -> Dict[str, Any]:
    """Adds the next question to chat history if available."""
    print("\n--- Executing ask_question_logic ---") # Debug print
    num_answers = len(state.get("answers", []))
    if num_answers < len(questions):
        question_text = questions[num_answers]
        # Add bot question to chat history only if it's not already the last message
        if not st.session_state.chat_history or \
           not (st.session_state.chat_history[-1].get("role") == "bot" and \
                st.session_state.chat_history[-1].get("content", "").endswith(question_text)) :
             st.session_state.chat_history.append({"role": "bot", "content": f"**Q{num_answers+1}:** {question_text}"})
    return state # State doesn't change just by asking a question

# Node 2: Collect the user's answer (Logic)
# This function is called *by the Streamlit flow* after input is received.
def collect_answer_logic(state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """Adds user answer to state and chat history."""
    print("\n--- Executing collect_answer_logic ---") # Debug print
    answer = user_input.strip()
    if answer:
        current_answers = state.get("answers", []).copy()
        current_answers.append(answer)
        # Add user answer to chat history
        st.session_state.chat_history.append({"role": "user", "content": answer})
        # Update state
        new_state = state.copy()
        new_state["answers"] = current_answers
        return new_state
    return state # Return original state if no input


# Conditional function: Check if interview complete (Logic)
# Used by Streamlit flow to decide next step.
def check_completion_condition(state: Dict[str, Any]) -> str:
    print("\n--- Executing check_completion_condition ---") # Debug print
    num_answers = len(state.get("answers", []))
    print(f"Answers collected: {num_answers}, Total questions: {len(questions)}")
    if num_answers >= len(questions):
        print("Condition met: Next step is Report")
        return "generate_report" # Return a string representing the *phase* or *next action*
    else:
        print("Condition not met: Next step is Ask")
        return "ask_question" # Return a string representing the *phase* or *next action*


# Node 3: Generate the final report (Logic)
def generate_report_logic(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generates report and adds to chat history."""
    print("\n--- Executing generate_report_logic ---") # Debug print
    answers = state.get("answers", [])
    if not answers:
        st.session_state.chat_history.append({"role": "bot", "content": "No answers collected. Cannot generate report."})
        return state # Return state even if no report generated

    formatted_answers = "\n".join(
        [f"Q{i+1}: {questions[i]}\nA{i+1}: {answers[i]}" for i in range(len(answers))]
    )
    prompt = f"""
You are an experienced HR specialist responsible for evaluating job candidates after structured interviews. Below are the candidate's responses to a series of interview questions. Your task is to write a well-structured, formal 5-paragraph evaluation report. This report should reflect the candidate's communication skills, professional experience, critical thinking, and cultural fit for the role.

Please ensure the report includes:

1. **Introduction** — Briefly introduce the candidate and the interview context.
2. **Strengths** — Highlight key strengths observed from the responses.
3. **Weaknesses or Areas for Improvement** — Mention any concerns or gaps noted during the interview.
4. **Overall Impression** — Summarize how the candidate presented themselves professionally.
5. **Recommendation** — Conclude with your recommendation on the candidate’s suitability for the role (e.g., move forward to next round, strong hire, needs more evaluation, etc.).

Interview Responses:
{formatted_answers}
"""

    try:
        report_content = llm.invoke(prompt).content
        st.session_state.chat_history.append({"role": "bot", "content": "### Final Evaluation Report:\n" + report_content})
    except Exception as e:
        st.session_state.chat_history.append({"role": "bot", "content": f"Error generating report: {e}"})

    return state # Return the state




workflow = StateGraph(dict) # State is a dictionary

# Add nodes using the logic functions
workflow.add_node("ask_question", ask_question_logic) # Node representing asking a question
workflow.add_node("collect_answer", collect_answer_logic) # Node representing collecting an answer
workflow.add_node("generate_report", generate_report_logic) # Node representing generating report

# Define the transitions - Represents the intended flow
workflow.set_entry_point("ask_question") # Start by asking

# From asking, the conceptual next step is collecting the answer
# In the UI, this transition happens after user input is received
workflow.add_edge("ask_question", "collect_answer")

# After collecting, check completion to decide next step
workflow.add_conditional_edges(
    "collect_answer", # Node coming FROM (after answer collected)
    check_completion_condition, # Function deciding next step based on state
    {                             # Map decision string to next node name
        "ask_question": "ask_question",
        "generate_report": "generate_report"
    }
)

# After generating report, interview ends
workflow.add_edge("generate_report", END)




# --- Streamlit App Layout and Interaction Flow (Manually managing steps) ---

st.title("🧠 AI Interview Chatbot")

# Display chat history FIRST. This loop runs on every rerun.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



current_graph_state = st.session_state.graph_state
num_answers = len(current_graph_state.get("answers", []))
total_questions = len(questions)

# Initialize the interview if it's not started yet
if not st.session_state.interview_started and not st.session_state.interview_complete:
    st.session_state.interview_started = True
    ask_question_logic(current_graph_state)
    st.rerun()

# Determine the current phase based on the number of answers
interview_is_complete = num_answers >= total_questions
current_phase = check_completion_condition(current_graph_state) # "ask_question" or "generate_report" based on answers

# --- Step 1: Display Questions or Report ---
if current_phase == "ask_question":
    if not interview_is_complete: # Ensure we don't ask more questions if complete
        # The first question is already added in the initialize section
        # Only call ask_question_logic after an answer has been submitted
        
        # Display the input box using st.chat_input
        user_answer = st.chat_input("Your Answer:", key="current_answer_input")

        # --- Step 2: Process Submitted Answer ---
        # This block executes on the rerun that happens *after* the user submits input via chat_input
        if user_answer:
            print(f"\n--- User submitted answer: {user_answer} ---")
            # Call the collect_answer logic to update the state and add to history
            st.session_state.graph_state = collect_answer_logic(current_graph_state, user_answer)
            
            # After collecting an answer, ask the next question immediately
            ask_question_logic(st.session_state.graph_state)
            
            # Force a rerun to re-evaluate the flow logic based on the updated state
            st.rerun()

elif current_phase == "generate_report":
     if not st.session_state.report_generated:
         print("\n--- Transitioning to generate_report phase ---")
         st.session_state.report_generated = True # Set flag immediately
         st.info("Generating interview report...")
         # Call the generate_report logic function
         generate_report_logic(current_graph_state)
         # Mark interview complete after report is generated
         st.session_state.interview_complete = True
         # Force a rerun to display the report and completion message
         st.rerun()


# --- Final Completion Message ---
# This appears below the input or report
if st.session_state.interview_complete:
    st.success("Interview completed!")
    st.info("Scroll up to see the complete interview and report.")


# --- Reset Button ---
# Place the reset button at the bottom
if st.button("Start New Interview"):
    print("\n--- Resetting interview state ---")
    # Reset all session state variables
    st.session_state.graph_state = {"answers": []}
    st.session_state.chat_history = []
    st.session_state.interview_complete = False
    st.session_state.report_generated = False
    st.session_state.interview_started = False  # Reset the interview_started flag
    
    # Clear the value associated with the chat_input key
    if "current_answer_input" in st.session_state:
         del st.session_state["current_answer_input"]

    st.rerun() # Force a rerun to restart the app flow from the beginning
