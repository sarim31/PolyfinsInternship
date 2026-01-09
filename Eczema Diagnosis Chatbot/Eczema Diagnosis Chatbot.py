import openai

# API key removed for security reasons

def ask_question(prompt):
    response = openai.Completion.create(
        engine="text-oobabooga-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=200,
        stop=None
    )
    return response['choices'][0]['text'].strip()

def main():

# Initial prompt and introduction
    prompt = """
    You are a chatbot designed to diagnose eczema based on the Hanifin-Rajka criteria. 
    The goal is to ask the user relevant questions to determine whether they have eczema or not. 
    The Hanifin-Rajka criteria require at least three major features and three minor features for a conclusive diagnosis.

Major features:
1. Pruritus (itching)
2. Typical morphology and distribution (flexural lichenification and linearity in adults; facial and extensor involvement in infants and children)
3. Chronic or chronically relapsing dermatitis
4. Personal or family history of atopy (asthma, allergic rhinitis, atopic dermatitis)

Minor features:
1. Ichthyosis, palmar hyperlinearity, or keratosis pilaris 
2. Xerosis (dry skin) 
3. Immediate (type I) skin test reactivity 
4. Elevated serum IgE level 
5. Early age of onset 
6. Tendency toward cutaneous infections (especially staphylococcal and herpes simplex) or impaired cell-mediated immunity 
7. Tendency toward nonspecific hand or foot dermatitis 
8. Nipple eczema 
9. Cheilitis (inflammation of the lips) 
10. Recurrent conjunctivitis 
11. Dennie-Morgan infraorbital fold (a fold or line in the skin below the lower eyelid) 
12. Keratoconus (a cone-shaped cornea) and anterior subcapsular cataracts 
13. Orbital darkening 
14. Facial pallor or facial erythema 
15. Pityriasis alba (patches of lighter skin, usually on the face) 

The confidence threshold for the final diagnosis is set at 0.8 or higher. The model will iteratively ask questions until it reaches sufficient confidence in the diagnosis (Yes/No).

Please answer the questions to the best of your ability. If you are unsure or your answer is invalid, incomplete, or unclear, the bot will ask for clarification or rephrasing.

---

"""

    # Iterative questioning
    confidence_threshold = 0.8
    current_confidence = 0.0

    while current_confidence < confidence_threshold:
        user_response = ask_question(prompt)
        print(f"Bot: {user_response}")

        # Analyze user response and update prompt
        prompt += f"\nUser: {user_response}\n"

        # Extract confidence level from the response
        current_confidence = float(response['choices'][0]['confidence'])

    # Final diagnosis
    final_diagnosis = "Yes" if current_confidence >= confidence_threshold else "No"
    print(f"The final diagnosis is: {final_diagnosis}")

if __name__ == "__main__":
    main()