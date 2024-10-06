from orchestrator import MainOrchestrator

main_orchestrator = MainOrchestrator()

question = "How do I know if I have glaucoma?"

conversation_id, response, evaluation_result, cost = main_orchestrator.run(
    question=question
)

print(f"Conversation ID: {conversation_id}")
print(f"Response: {response}")
print(f"Evaluation Result: {evaluation_result}")
print(f"Cost: {cost}")
