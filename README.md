# Market Sentiment Analyzer  

## Project Description  
The **Stock Market Sentiment Analyzer** is a Python-based application that helps users determine the **current market sentiment** for any stock.  
It fetches **real-time news articles** using **NewsAPI**, processes them using **machine learning-based sentiment analysis**, and presents the results in a **Tkinter-based UI**.  

## Features  
- **Fetches Stock News**: Retrieves the latest news articles for any given stock using **NewsAPI**.  
- **Machine Learning Sentiment Analysis**: Uses a **Naive Bayes model**, trained on a **Kaggle dataset**, to classify news sentiment as **Positive, Negative, or Neutral**.  
- **User-Friendly Interface**: Built with **Tkinter**, the UI provides a clean and professional experience for users.  

---

### How It Works
User enters a stock ticker (e.g., AAPL for Apple, TSLA for Tesla).
Application fetches real-time news about the stock.
Each news article is analyzed using a Naive Bayes sentiment model trained on a Kaggle dataset.
Sentiment result (Positive/Negative/Neutral) is displayed in the UI.
The top news articles are listed in a scrollable section for user reference.

---

##  Tech Stack  
- **Python** 🐍  
- **Tkinter** (GUI framework)  
- **Scikit-learn** (for machine learning)  
- **Pandas** (for data handling)  
- **NewsAPI** (for fetching real-time stock news)  
- **NLTK** (for text preprocessing)  

---

## Installation Guide  

### **1️⃣ Clone the Repository**  

git clone https://github.com/your-username/Stock-Market-Sentiment-Analyzer.git
cd Stock-Market-Sentiment-Analyzer

### **2️⃣ Install Dependencies**
Make sure you have Python installed. Then, install the required libraries:
sh
Copy
Edit
pip install -r requirements.txt

### **3️⃣ Set Up API Keys**
Get a NewsAPI key from NewsAPI.
Open the script and replace "your_newsapi_key" with your actual API key.

### **4️⃣ Run the Application**
sh
Copy
Edit
python app.py

