from flask import Flask,request,render_template,url_for


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['GET','POST'])

def predict():

    import pandas as pd
    import numpy as np
    import random



    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split


    # In[9]:


    urls_data = pd.read_csv("/Users/saksh/OneDrive/Desktop/Malicious-Webpage-detection-master/data.csv")
    print(urls_data)

    # In[10]:


    type(urls_data)


    # In[11]:


    urls_data.head()


    # In[25]:


    def makeTokens(f):
        tkns_BySlash = str(f.encode('utf-8')).split('/')
        total_Tokens = []
        for i in tkns_BySlash:
            tokens = str(i).split('-')
            tkns_ByDot = []
            for j in range(0,len(tokens)):
                temp_Tokens = str(tokens[j]).split('.')
                tkns_ByDot = tkns_ByDot + temp_Tokens
            total_Tokens = total_Tokens + tokens + tkns_ByDot
        total_Tokens = list(set(total_Tokens))
        if 'com' in total_Tokens:
            total_Tokens.remove('com')
        return total_Tokens




    # In[26]:


    y = urls_data["label"]


    # In[27]:


    url_list = urls_data["url"]


    # In[28]:


    vectorizer = TfidfVectorizer(tokenizer = makeTokens)


    # In[29]:


    X = vectorizer.fit_transform(url_list)


    # In[31]:


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # In[32]:


    logit = LogisticRegression()
    logit.fit(X_train, y_train)



    print("Accuracy ",logit.score(X_test, y_test))
    if request.method=='POST':
        comment=request.form.get('comment')
        X_predict1=[comment]
        predict1 = vectorizer.transform(X_predict1)
        New_predict1 = logit.predict(predict1)
        new = New_predict1.tolist()
        new1 = " ".join(str(x) for x in new)
        return render_template('result.html', prediction=new1)
    return render_template('result.html')
        #new = New_predict1.tolist()
        #new1 = " ".join(str(x) for x in new)
    #return render_template('result.html',prediction=new1)





if __name__ == '__main__':
    app.run(debug=True)
