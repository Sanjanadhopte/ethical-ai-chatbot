import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Expanded FAQ knowledge base with detailed answers
faq = {
    "what is ai ethics": (
        "AI ethics is a set of principles guiding AI development and use, "
        "covering fairness, accountability, transparency, privacy, and harm prevention."
    ),
    "why do we need ethical concerns with ai": (
        "Ethical concerns are crucial in AI to prevent bias, protect privacy, "
        "ensure fairness, and build trust in AI systems."
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
    # Add other Q&A as needed
}

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
        return "Sorry, I couldn't find a detailed answer for that. Please try rephrasing your question."

# Bias and fairness training module text
def bias_and_fairness_module():
    st.write("### Module: Bias and Fairness in AI")
    st.write("Bias in AI can lead to unfair treatment of individuals or groups due to data or algorithmic issues.")
    st.write("Example: An AI hiring system favoring certain demographics unfairly.")
    answer = st.radio("Quiz: Is an AI that declines applicants based on gender bias-free?", ('yes', 'no'))
    if answer:
        if answer == 'no':
            st.success("Correct! That AI exhibits bias and fairness concerns.")
        else:
            st.error("Incorrect. Bias is a critical ethical issue to address.")
    reflection = st.text_area("Reflection: Have you encountered or observed AI bias? How did it affect outcomes?")
    if reflection:
        st.info("Thank you for sharing! Awareness helps build fair AI systems.")

# Streamlit app layout
st.title("Ethical AI Training & Assistant")

menu = st.sidebar.selectbox("Choose an option", ["Training Module", "AI Assistant Chat"])

if menu == "Training Module":
    bias_and_fairness_module()

if menu == "AI Assistant Chat":
    user_input = st.text_input("Ask me about AI ethics, cybersecurity, or digital literacy:")
    if user_input:
        response = semantic_search_response(user_input)
        st.markdown("**Assistant:** " + response)
