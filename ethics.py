from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAQ knowledge base (unchanged, as in your original code)
faq = {
    # ... (use your full FAQ dictionary here, omitted for brevity)
    "what is AI ethics": (
        "AI ethics is a set of principles guiding AI development and use, "
        "covering fairness, accountability, transparency, privacy, and harm prevention."
    ),
    "how to recognize phishing email": (
        "Recognize phishing by checking suspicious sender addresses, "
        "looking for urgent or alarming language, and inspecting links without clicking."
    ),
     "what is digital literacy": (
        "Digital literacy involves skills to find, evaluate, and communicate information "
        "effectively using digital technologies."
    ),
  "what is cybersecurity": (
        "Cybersecurity is the practice of protecting systems, networks, and data from "
        "cyber threats like hacking, phishing, and malware."
    ),
    "what is two factor authentication": (
        "Two-factor authentication (2FA) adds an extra layer of security by requiring "
        "a password plus a second method, like a code sent to your phone."
    ),
    "what is data privacy": (
        "Data privacy refers to controlling how personal information is collected, "
        "stored, and shared, ensuring users’ data is kept safe and confidential."
    ),
    "what is machine learning": (
        "Machine learning is a branch of AI where systems learn patterns from data "
        "and improve performance without being explicitly programmed."
    ),
    "what are strong passwords": (
        "Strong passwords are long, unique, and include a mix of uppercase letters, "
        "lowercase letters, numbers, and special symbols."
    ),
    "what is plagiarism in digital world": (
        "Digital plagiarism is copying online content, code, or ideas without proper "
        "credit, violating academic or ethical guidelines."
    ),
    "what is fake news": (
        "Fake news is false or misleading information spread online to misinform or "
        "influence people, often through social media."
    ),
    "how to stay safe online": (
        "Stay safe by using strong passwords, enabling 2FA, avoiding suspicious links, "
        "and keeping your software updated."
    ),
    "what is digital footprint": (
        "A digital footprint is the trail of data you leave online, including social media "
        "posts, browsing history, and interactions."
    ),
     "how can ai help students in learning": (
        "AI can personalize learning, provide instant feedback, recommend study materials, "
        "and help students understand complex topics more easily."
    ),
    "what digital skills are important for students today": (
        "Students need skills like online research, data analysis, coding basics, "
        "digital communication, cybersecurity awareness, and AI literacy."
    ),
    "how to use ai tools responsibly as a student": (
        "Students should use AI for learning support, not shortcuts. They must cross-check "
        "facts, avoid plagiarism, and respect data privacy."
    ),
    "why is digital literacy important for students": (
        "Digital literacy helps students find reliable information, stay safe online, "
        "collaborate effectively, and prepare for technology-driven careers."
    ),
    "what is the role of ai in education": (
        "AI in education supports adaptive learning platforms, automated grading, "
        "virtual tutors, and helps teachers focus on personalized teaching."
    ),
    "how does digital literacy help in jobs": (
        "Employers value digital literacy as it shows the ability to use technology, "
        "analyze information, and adapt to digital tools in the workplace."
    ),
    "what is online collaboration": (
        "Online collaboration is working with others using digital tools like Google Docs, "
        "Microsoft Teams, or Zoom to share ideas and complete tasks together."
    ),
    "what is algorithmic bias": (
        "Algorithmic bias happens when an AI system produces unfair or discriminatory "
        "results because of biased data or flawed design."
    ),
    "how can ai affect fairness": (
        "AI can affect fairness if it treats groups differently, for example in hiring, "
        "credit scoring, or facial recognition, leading to discrimination."
    ),
    "what is transparency in ai": (
        "Transparency in AI means making AI decisions and processes understandable "
        "so people know how and why an outcome was reached."
    ),
    "why is accountability important in ai": (
        "Accountability ensures that humans, not machines, are responsible for the "
        "consequences of AI decisions, especially in sensitive areas like healthcare or law."
    ),
    "what is explainable ai": (
        "Explainable AI (XAI) refers to AI systems designed to clearly explain their "
        "decisions and actions in human-understandable ways."
    ),
    "what are the risks of unethical ai": (
        "Unethical AI can spread misinformation, invade privacy, reinforce stereotypes, "
        "reduce trust, and cause real harm in society."
    ),
    "what is responsible ai": (
        "Responsible AI means developing and using AI in ways that are ethical, "
        "transparent, fair, safe, and aligned with human values."
    ),
    "why is privacy important in ai": (
        "Privacy is important because AI often uses personal data. Protecting privacy "
        "prevents misuse, identity theft, and loss of trust."
    ),
    "what is the role of governments in ai ethics": (
        "Governments create rules and regulations to make sure AI is used responsibly, "
        "protects citizens’ rights, and avoids harmful impacts."
    ),
     "what is ethics": (
        "Ethics is the study of what is right and wrong in human behavior. "
        "It guides people to act fairly, honestly, and responsibly."
    ),
    "why are ethics important": (
        "Ethics are important because they help people make good decisions, "
        "build trust, and live together in a fair society."
    ),
    "what are the rules of ethics": (
        "Common ethical rules include honesty, fairness, respecting others, "
        "taking responsibility for actions, and avoiding harm."
    ),
    "what is the difference between ethics and laws": (
        "Laws are rules made by governments that people must follow. "
        "Ethics are moral principles that guide behavior, even when no law exists."
    ),
    "what are examples of ethics in daily life": (
        "Examples include telling the truth, not cheating in exams, "
        "helping others, and respecting different opinions."
    ),
    "what are professional ethics": (
        "Professional ethics are rules of good behavior at work, "
        "like honesty, respect, responsibility, and not misusing power."
    ),
    "what is the golden rule in ethics": (
        "The golden rule is 'treat others the way you want to be treated,' "
        "a simple guide for ethical behavior."
    ),
    "what happens if people do not follow ethics": (
        "If people ignore ethics, it can lead to cheating, corruption, "
        "loss of trust, and unfair treatment in society."
    ),
    "how can students practice ethics": (
        "Students can practice ethics by being honest in exams, respecting teachers and classmates, "
        "avoiding plagiarism, and using technology responsibly."
    ),
      "what is digital literacy with example": (
            "Digital literacy is the ability to use digital tools effectively. "
            "For example, knowing how to find reliable information online instead of believing fake news."
   ),
    "what is digital footprint with example": (
            "A digital footprint is the record of what you do online. "
            "For example, your social media posts, comments, and browsing history leave a digital footprint."
    ),
    "what is plagiarism in digital world with example": (
         "Digital plagiarism is copying online content without credit. "
        "For example, copy-pasting an article for an assignment without mentioning the source."
    ),
    "how to stay safe online with example": (
        "Staying safe online means protecting your data and privacy. "
        "For example, using strong passwords and not sharing your bank details on suspicious websites."
    ),
    "what is fake news with example": (
        "Fake news is false information spread online. "
        "For example, a social media post claiming a celebrity died when they are actually alive."
    ),
    "what are examples of ethics in daily life": (
        "Examples include telling the truth, not cheating in exams, "
        "helping others, and respecting different opinions."
    ),
        "what is ai ethics with example": (
        "AI ethics is about using AI responsibly and fairly. "
        "For example, an AI hiring system should treat all candidates equally "
        "without discriminating based on gender or race."
    ),
    "what is algorithmic bias with example": (
        "Algorithmic bias happens when AI gives unfair results due to biased data. "
        "For example, a facial recognition system that works better on lighter skin tones "
        "but makes errors on darker skin tones."
    ),
    "what is transparency in ai with example": (
        "Transparency means explaining how AI makes decisions. "
        "For example, a loan approval AI should explain why someone was rejected, "
        "not just give a 'yes' or 'no' answer."
    ),
    "what is responsible ai with example": (
        "Responsible AI means developing AI that is safe and fair. "
        "For example, medical AI tools should be carefully tested before being used on patients."
    ),
    "How can we stop ethical fraud":(
    "To stop ethical fraud, we need clear ethical rules,"
    "regular training, strict monitoring, safe reporting systems, and strong accountability,"
     "while promoting a culture of honesty and fairness"
    ),
    "How can users uphold ethical responsibility while interacting with AI systems":(
    "User responsibility in AI ethics involves ensuring that technology is used in a transparent, fair, and "
    "accountable manner. Users must actively safeguard privacy, prevent misuse, and promote inclusivity and "
    "trust when interacting with AI systems. Ethical use also requires understanding AI’s limitations," 
    "maintaining human oversight, and ensuring AI decisions align with human values and social well-being")
    
    # ... (include all your FAQ question-answer pairs)
}

# Precompute embeddings for FAQ questions
faq_questions = list(faq.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

def semantic_search_response(user_query, threshold=0.6):
    user_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, faq_embeddings)[0]
    top_idx = np.argmax(cosine_scores)
    top_score = cosine_scores[top_idx].item()
    if top_score >= threshold:
        return faq[faq_questions[top_idx]]
    else:
        return None

# --- Ethical training modules with stepwise progression ---

def stepwise_module(steps):
    idx = 0
    while idx < len(steps):
        step = steps[idx]
        print(step["text"])
        if step["action"] == "continue":
            input("Press Enter to continue...")
        elif step["action"].startswith("quiz"):
            answer = input("Your answer: ").strip().lower()
            feedback = step["feedback"].get(answer, "Please answer 'yes' or 'no'.")
            print(feedback)
        elif step["action"] == "reflect":
            input("Share your thoughts (press Enter to continue): ")
        # Automatically continue to next step
        idx += 1

def ethical_training_module_bias_and_fairness():
    steps = [
        {
            "text": "\n--- Module: Bias and Fairness in AI ---\n"
                    "Why does fairness matter in AI?\n"
                    "Bias in AI can lead to unfair treatment of individuals or groups due to data or algorithmic issues.\n"
                    "Example: An AI hiring system favoring certain demographics unfairly.\n",
            "action": "continue"
        },
        {
            "text": "Quiz: Is an AI that declines applicants based on gender bias-free? (yes/no)",
            "action": "quiz1",
            "feedback": {
                "no": "Correct! That AI exhibits bias and fairness concerns.",
                "yes": "Incorrect. Bias is a critical ethical issue to address."
            }
        },
        {
            "text": "Reflection: Have you encountered or observed AI bias? How did it affect outcomes?",
            "action": "reflect"
        },
        {
            "text": "\nRemember, mitigating bias ensures fairness and trust in AI systems.",
            "action": "continue"
        }
    ]
    stepwise_module(steps)

def ethical_training_module_privacy_and_security():
    steps = [
        {
            "text": "\n--- Module: Privacy and Security in AI ---\n"
                    "Why is privacy important?\n"
                    "AI systems often handle personal data. If not protected, this data can be misused or stolen.\n"
                    "Example: A chatbot storing private conversations without user consent.\n",
            "action": "continue"
        },
        {
            "text": "Quiz: Should AI systems collect only the data that is necessary? (yes/no)",
            "action": "quiz2",
            "feedback": {
                "yes": "Correct! Only necessary data should be collected to reduce privacy risks.",
                "no": "Incorrect. Minimizing data collection is a privacy best practice."
            }
        },
        {
            "text": "Reflection: Why do you think data privacy is especially important with AI systems?",
            "action": "reflect"
        },
        {
            "text": "\nRemember, privacy and security are vital for building trust in AI.",
            "action": "continue"
        }
    ]
    stepwise_module(steps)

def ethical_training_module_transparency_and_explainability():
    steps = [
        {
            "text": "\n--- Module: Transparency and Explainability ---\n"
                    "Why does transparency matter?\n"
                    "Users should know how AI makes decisions to build trust and accountability.\n"
                    "Example: A loan approval AI should explain why an application was rejected.\n",
            "action": "continue"
        },
        {
            "text": "Quiz: Should AI decisions always be understandable to humans? (yes/no)",
            "action": "quiz3",
            "feedback": {
                "yes": "Correct! Understandability is key for trust, accountability, and error detection.",
                "no": "Incorrect. AI decisions should be explainable for transparency."
            }
        },
        {
            "text": "\nTransparency and explainability are crucial for responsible AI adoption.",
            "action": "continue"
        }
    ]
    stepwise_module(steps)

def ethical_training_module_accountability():
    steps = [
        {
            "text": "\n--- Module: Accountability in AI ---\n"
                    "Why is accountability important?\n"
                    "Humans, not machines, are responsible for the outcomes of AI decisions.\n"
                    "Example: A self-driving car accident must be investigated with human responsibility.\n",
            "action": "continue"
        },
        {
            "text": "Quiz: Should AI be allowed to operate without human oversight? (yes/no)",
            "action": "quiz4",
            "feedback": {
                "no": "Correct! Human oversight is necessary for accountable and safe AI use.",
                "yes": "Incorrect. Lack of human oversight increases risk and reduces accountability."
            }
        },
        {
            "text": "\nAccountability ensures that AI serves and protects human interests.",
            "action": "continue"
        }
    ]
    stepwise_module(steps)

def ethical_training_module_misinformation():
    steps = [
        {
            "text": "\n--- Module: Misinformation and AI ---\n"
                    "Why is misinformation a concern?\n"
                    "AI can spread fake news or misleading content if not carefully managed.\n"
                    "Example: Deepfake videos used to manipulate public opinion.\n",
            "action": "continue"
        },
        {
            "text": "Quiz: Should we verify AI-generated content before sharing? (yes/no)",
            "action": "quiz5",
            "feedback": {
                "yes": "Correct! Verifying content is key to preventing misinformation.",
                "no": "Incorrect. Always verify before sharing to reduce risks."
            }
        },
        {
            "text": "\nVigilance against misinformation helps protect digital spaces and public trust.",
            "action": "continue"
        }
    ]
    stepwise_module(steps)

def ethical_training_menu():
    while True:
        print("\n--- Ethical Training Modules ---")
        print("1. Bias and Fairness in AI")
        print("2. Privacy and Security in AI")
        print("3. Transparency and Explainability")
        print("4. Accountability in AI")
        print("5. Misinformation and AI")
        print("6. Return to main menu")
        choice = input("Choose a training module (1-6): ").strip()
        if choice == '1':
            ethical_training_module_bias_and_fairness()
        elif choice == '2':
            ethical_training_module_privacy_and_security()
        elif choice == '3':
            ethical_training_module_transparency_and_explainability()
        elif choice == '4':
            ethical_training_module_accountability()
        elif choice == '5':
            ethical_training_module_misinformation()
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please select a number from 1 to 6.")

def main():
    print("Welcome to the Ethical AI Training & Assistant Demo!")
    while True:
        print("\nChoose an option:")
        print("1. Take Ethical Training Module")
        print("2. Chat with Ethical AI Assistant(Ask question)")
        print("3. Exit")
        choice = input("Your choice (1/2/3): ").strip()
        if choice == '1':
            ethical_training_menu()
        elif choice == '2':
            print("Start chatting with the AI assistant (type 'exit' to return to menu).")
            while True:
                user_query = input("You: ").strip()
                if user_query.lower() in ["exit", "quit"]:
                    print("Exiting assistant chat.")
                    break
                response = semantic_search_response(user_query)
                if response:
                    print("Assistant:", response)
                else:
                    print(
                        "Assistant: Sorry, I couldn't find a detailed answer for that. "
                        "Could you please rephrase or ask something else?"
                    )
        elif choice == '3':
            print("Thank you for participating! Goodbye.")
            break
        else:
            print("Invalid choice, please select 1, 2 or 3.")

if __name__ == "__main__":
    main()
