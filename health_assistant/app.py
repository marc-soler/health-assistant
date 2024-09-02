import streamlit as st
from orchestrator import MainOrchestrator

# Streamlit App
st.title("Health Assistant Chatbot")

# Initialize the Main Orchestrator
main_orchestrator = MainOrchestrator()

# User query input
question = st.text_input("Enter your health-related question:")

if st.button("Get Answer"):
    if question:
        context = "The patient is a 45-year-old male with a history of hypertension."

        # Run the main orchestration process
        conversation_id, response, evaluation_result, cost = main_orchestrator.run(
            context=context, question=question
        )

        # Display LLM response
        st.subheader("Response:")
        st.write(response)

        # Display evaluation result (optional)
        st.subheader("Evaluation Result:")
        st.write(evaluation_result)

        # Display cost
        st.write(f"OpenAI cost for this query: ${cost:.6f}")

        # Feedback section
        st.subheader("Provide Feedback:")
        feedback = st.radio("Was this answer helpful?", ("Like", "Dislike"))
        additional_feedback = st.text_area("Any additional feedback? (optional)")

        if st.button("Submit Feedback"):
            main_orchestrator.feedback_orchestrator.submit_feedback(
                conversation_id=conversation_id,
                feedback=feedback,
                additional_feedback=additional_feedback,
            )
            st.success("Thank you for your feedback!")

    else:
        st.error("Please enter a question.")
