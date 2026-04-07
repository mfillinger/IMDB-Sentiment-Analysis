import streamlit as st
import joblib

#load saved model and vectorizer
model= joblib.load("best_sentiment_model.pkl")
vectorizer= joblib.load("tfidf_vectorizer.pkl")

st.title("IMDB Sentiment Analysis")
st.caption("Model: TF-IDF + Logistic Regression | Dataset: IMDB 50K Reviews")
st.write("Enter a movie review and the model will predict the sentiment.")

review= st.text_area("Movie Review")

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        review_tfidf= vectorizer.transform([review])

        prediction= model.predict(review_tfidf)
        probability= model.predict_proba(review_tfidf)

        confidence= probability.max()
        st.divider()
        if prediction[0] == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

        st.caption(f"Confidence: {confidence:.2f}")
