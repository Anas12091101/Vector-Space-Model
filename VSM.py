from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import *
import collections
import re
import math
import numpy as np
from numpy.linalg import norm

def stopwords_init():
    file = open("Stopword-List.txt","r")
    stopwards = file.read().split()
    return stopwards

def load_data():
    terms = []
    docs = []
    stopwards = stopwords_init()
    # stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    for i in range(448):
        file = open(f"Abstracts/{i+1}.txt","r",encoding="latin1")
        d={}
        for line in file:
            for word in re.findall(r"[\w']+",line):  
                word=word.replace("'","")
                # stemmed = stemmer.stem(word)
                stemmed = lemmatizer.lemmatize(word.lower())
                if word.lower() not in stopwards and stemmed not in terms:
                    terms.append(stemmed)    
                    d[stemmed] = 1
                elif stemmed in terms and stemmed in d.keys():
                    d[stemmed] += 1
                elif stemmed in terms and stemmed not in d.keys():
                    d[stemmed] = 1
        docs.append(d)
    return terms,docs

def calc_df(terms,docs):
    df = {}
    for term in terms:
        for doc in docs:
            if term in doc.keys():
                if term in df.keys():
                    df[term] += 1
                else:
                    df[term] = 1
    return df

def populate_doc_vector(terms,docs,df):
    doc_vector = []
    df_val = list(df.values())
    for i in range(len(docs)):
        d = [0]*len(terms)
        doc_vector.append(d)

    for idx,doc in enumerate(doc_vector):
        for i,term in enumerate(terms):
            if term in docs[idx].keys():
                doc[i] = (docs[idx][term]/len(docs[idx])) * math.log((len(docs)/df_val[i]),10)
    return doc_vector

def calculate_cosSim(doc_vector,query_vector):
    results = {}
    qv = np.array(query_vector)
    for idx,dv in enumerate(doc_vector):
        dv_arr = np.array(dv)
        results[f"{idx+1}"] = np.dot(qv,dv_arr)/(norm(qv)*norm(dv_arr))
    print(results)
    return results


def processquery(query,terms,docs,df,doc_vector):
    lemmatizer = WordNetLemmatizer()
    for idx,word in enumerate(query):
        query[idx] = lemmatizer.lemmatize(word)
    print(query)
    count = collections.Counter(query)
    query_vector = [0]*len(terms)
    for idx,term in enumerate(terms):
        if term in query:
            query_vector[idx] = (count[term]/len(query)) * math.log((len(docs)/df[term]),10)  # (normalized tf) * (normalized idf)
    
    print(query_vector)
    res = calculate_cosSim(doc_vector,query_vector)
    fin_res = [k for k,r in res.items() if r > 0.001 ]  # list containing all the terms with cos similarity greater than 0.001(alpha)
    return fin_res

def inputqueryGUI(terms,docs,df,doc_vector):
    #a method to get input queries from user and passing to the main process query fn.
    def execute():
        label2["text"]=""
        query=str(entry.get())
        query=query.lower().split()
        doc_list=processquery(query,terms,docs,df,doc_vector)
        if doc_list:
            label2["fg"]="purple"
            i=0
            
            #for loop for mantaining the no of docs in one output line.
            for doc in doc_list:
                if i==20:
                    label2["text"]+="\n"+str(doc)+" "
                    i=0
                else:
                    label2["text"]+=str(doc)+" "
                i=i+1
        else:
            label2["fg"]="red"
            label2["text"]="Sorry didn't find the document. Please enter correct query"
        doc_list.clear()

    #GUI design line[242-266]
    Font_tuple = ("Comic Sans MS", 20, "bold")
    window=tk.Tk()
    window.title("Vector Space Model( By 19k-1534)")
    window.geometry("1000x800")
    window.rowconfigure(0, minsize=500, weight=1)
    window.columnconfigure([0, 1, 2], minsize=500, weight=1)
    frame1=tk.Frame(master=window,width=200,height=70,bg="#4a0f61")
    frame1.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
    label1=tk.Label(master=frame1,font=Font_tuple,text="Welcome!",bg="#4a0f61",fg="white")
    label1.pack()
    frame=tk.Frame(master=window,width=100,height=70,bg="white")
    frame.place(relx=.5, rely=.5,anchor= CENTER)
    frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
    label1=tk.Label(master=frame,text="Vector Space Model",bg="white",fg="purple",font=Font_tuple)
    label = tk.Label(master=frame,text="Enter your query",bg="white",fg="purple")
    entry=tk.Entry(master=frame,width=40)
    btn=tk.Button(master=frame,bg="purple",fg="white",text="Enter",command=execute)
    label1.pack()
    label.pack()
    entry.pack()
    btn.pack()
    label2 = Label(frame,font=('"Comic Sans MS" 8 bold'),width=200,bg="white",fg="purple",text="waiting for your query :)")
    label2.pack(pady=20)
    window.mainloop() 



def main():
    terms,docs = load_data()
    terms = sorted(terms)
    df = calc_df(terms,docs)
    doc_vector = populate_doc_vector(terms,docs,df)
    inputqueryGUI(terms,docs,df,doc_vector)


if __name__ == "__main__":
    main()