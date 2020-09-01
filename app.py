{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [31/Aug/2020 23:32:08] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [31/Aug/2020 23:32:08] \"\u001b[33mGET /static/css/styles.css HTTP/1.1\u001b[0m\" 404 -\n",
      "127.0.0.1 - - [31/Aug/2020 23:32:23] \"\u001b[37mPOST /predict HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [31/Aug/2020 23:32:23] \"\u001b[33mGET /static/css/styles.css HTTP/1.1\u001b[0m\" 404 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, redirect, url_for, render_template\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import LinearSVC\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "        return render_template('home.html')\n",
    "\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "    data = pd.read_excel('input_to_deploy.xlsx')\n",
    "    \n",
    "    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')   \n",
    "    features = tfidf.fit_transform(data.desc_no_stopword).toarray()\n",
    "    labels = data.group\n",
    "    \n",
    "    X_train, X_test, y_train, y_test = train_test_split(features, labels,test_size=0.2, random_state = 0, stratify=data['group'])\n",
    "    \n",
    "    clf = LinearSVC()\n",
    "    clf.fit(X_train,y_train)\n",
    "    clf.score(X_test,y_test)\n",
    "    \n",
    "    if request.method == 'POST':\n",
    "        message = request.form['message']\n",
    "        data = [message]\n",
    "        vect = tfidf.transform(data).toarray()\n",
    "        my_prediction = clf.predict(vect)\n",
    "    return render_template('result.html',prediction = my_prediction)    \n",
    "\n",
    "if __name__ == '__main__':\n",
    "     app.run(debug=True, use_reloader=False)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
