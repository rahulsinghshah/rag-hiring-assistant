def generate_answer(query, retrieved_docs):
    if not retrieved_docs:
        return "No matching candidates found."

    if "error" in retrieved_docs[0]:
        return f"Error: {retrieved_docs[0]['error']}"

    response = "Top Matching Candidates:\n\n"

    for i, doc in enumerate(retrieved_docs):
        response += f"Candidate {i+1}:\n"
        response += f"- Source: {doc['source']}\n"
        response += f"- Skills: {doc['candidate_skills']}\n"
        response += f"- Experience: {doc['candidate_experience']} years\n"
        response += f"- Final Score: {doc['final_score']}\n"
        response += f"- Why Selected: Matches query skills and experience\n\n"

    response += "Conclusion:\nCandidates ranked using semantic + structured + hybrid scoring."

    return response