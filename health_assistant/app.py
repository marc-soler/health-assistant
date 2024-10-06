import streamlit as st
from orchestrator import MainOrchestrator

# Initialize the Main Orchestrator
rag_orchestrator = MainOrchestrator()

# Streamlit App
st.title("Health Assistant Chatbot")


def show_feedback_section(conversation_id):
    st.subheader("Provide Feedback:")
    feedback = st.radio("Was this answer helpful?", ("Like", "Dislike"))
    additional_feedback = st.text_area("Any additional feedback? (optional)")

    if st.button("Submit Feedback"):
        # Submit feedback logic
        rag_orchestrator.feedback_orchestrator.submit_feedback(
            conversation_id=conversation_id,
            feedback=feedback,
            additional_feedback=additional_feedback,
        )
        st.success("Thank you for your feedback!")


def display_response(question):
    with st.spinner("Fetching answer..."):
        # Run the main orchestration process
        conversation_id, response, evaluation_result, cost = rag_orchestrator.run(
            question=question
        )

    # Display LLM response
    st.subheader("Response:")
    st.write(response)

    # Optionally display the evaluation result and cost in expandable sections
    with st.expander("Evaluation Result"):
        st.write(evaluation_result)

    with st.expander("Query Cost"):
        st.write(f"OpenAI cost for this query: ${cost:.6f}")

    # Show feedback section
    show_feedback_section(conversation_id)


# Main input for user question
question = st.text_input("Enter your health-related question:")

# Show loading spinner when user clicks the "Get Answer" button
if st.button("Get Answer"):
    if question:
        display_response(question)
    else:
        st.error("Please enter a question.")
