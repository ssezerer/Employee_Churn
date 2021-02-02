import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

html_temp = """
<div style="background-color:yellow;padding:4px">
<h2 style="color:blue;text-align:center;"> <b>EMPLOYEE CHURN PREDICTION</b> </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
my_slot1 = st.empty()
my_slot1.text('')

from PIL import Image
im = Image.open("employee_churn.jpeg")
st.image(im, use_column_width = True)

html_temp = """
<div style="background-color:green;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML App </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
my_slot2 = st.empty()
my_slot2.text('')

model = pickle.load(open("rf_imp_model", "rb"))
features=pickle.load(open("features.pkl", "rb"))
df_imp=pickle.load(open("df_imp.pkl", "rb"))
X_imp=pickle.load(open("X_imp.pkl", "rb"))

def random_customer():
    df_sample=X_imp.sample(number1)
    #df_5=df_sample.copy()
    #df_2=pd.get_dummies(df_sample).reindex(columns=features, fill_value=0)
    prediction=model.predict_proba(df_sample)
    result=model.predict(df_sample)
    df_sample["Churn Probability"]= prediction[:,1]
    df_sample["Churn Probability"]=df_sample["Churn Probability"].apply(lambda x: round(x,2))
    df_sample["result"]=result
    #df_sample["result"]=df_sample["result"].apply(lambda x: round(x))
    df_sample["result"]=df_sample["result"].apply(lambda x: "Churn" if x==1 else "Retain")
    a=df_sample.result.value_counts().values[0]
    b=df_sample.result.value_counts().values[1]
    fig, ax= plt.subplots()
    ax.bar(df_sample.result.value_counts().index, df_sample.result.value_counts())
    ax.set_title("Distribution of the Randomly Selected {} Customers".format(number1))
    return st.success("Number of the customers that will retain: {}".format(a)), st.warning("Number of the customers that will churn:{}".format(b)), st.pyplot(fig), st.table(df_sample)
            
def churn_customers():
    df_sample=X_imp.copy()
    #df_2=pd.get_dummies(df_sample).reindex(columns=features, fill_value=0)
    prediction=model.predict_proba(df_sample)
    result=model.predict(df_sample)
    df_sample["Churn Probability"]= prediction[:,1]
    df_sample["Churn Probability"]=df_sample["Churn Probability"].apply(lambda x: round(x,2))
    df_sample["result"]=result
    #df_sample["result"]=df_sample["result"].apply(lambda x: round(x))
    df_sample["result"]=df_sample["result"].apply(lambda x: "Churn" if x==1 else "Retain")
    df_sample=df_sample.sort_values(by="Churn Probability", ascending=False).head(number2)
    return st.table(df_sample)
    
def loyal_customers():
    df_sample=X_imp.copy()
    # df_2=pd.get_dummies(df_sample).reindex(columns=features, fill_value=0)
    prediction=model.predict_proba(df_sample)
    result=model.predict(df_sample)
    df_sample["Churn Probability"]= prediction[:,1]
    df_sample["Churn Probability"]=df_sample["Churn Probability"].apply(lambda x: round(x,2))
    df_sample["result"]=result
    #df_sample["result"]=df_sample["result"].apply(lambda x: round(x))
    df_sample["result"]=df_sample["result"].apply(lambda x: "Churn" if x==1 else "Retain")
    df_sample=df_sample.sort_values(by="Churn Probability").head(number)
    return st.table(df_sample)

#satisfaction_level, number_project, time_spend_company, average_monthly_hours, last_evaluation

if st.checkbox("Churn Probability of Single Employee"):
    
    satisfaction_level=st.sidebar.slider("Satisfaction level of employee", 0.0, 1.0, step=0.01)
    number_project=st.sidebar.slider("Number of projects conducted by employee", 2, 7, step=1)
    time_spend_company=st.sidebar.slider("Spending time of employee in company", 1, 10, step=1)
    average_monthly_hours=st.sidebar.slider("Average monthly working hours of employee", 95, 310, step=5)
    last_evaluation=st.sidebar.slider("Last evaluation of employee", 0.0, 1.0, step=0.01)

    my_dict = {'satisfaction_level':satisfaction_level,
           'number_project':number_project,
           'time_spend_company':time_spend_company,
           'average_montly_hours':average_monthly_hours,
           'last_evaluation':last_evaluation}
 
    if st.sidebar.checkbox("Okey"):
        single = pd.DataFrame.from_dict([my_dict], orient="columns")
        # X = pd.get_dummies(df).reindex(columns=features, fill_value=0)
        prediction = round(model.predict_proba(single)[:,-1][0]*100,2)
        result = "Churn" if model.predict(single)==1 else "Retain"
        st.markdown("### The features for Churn Prediction is below")
        st.table(single.head())
        st.markdown("### Press predict if features is okey")
        if st.button("Predict"):
            if result == "Churn":
                st.warning("Churn prediction is {}, Churn probability of the employee is %{}".format(result, prediction))
            else:
                st.success("Churn prediction is {}, Churn probability of the employee is %{}".format(result, prediction))

elif st.checkbox("Churn Probability of Randomly Selected Employees"):
    st.subheader("How many employees to select?")
    st.text("Please select the number of employees")
    number1=st.selectbox("", (5,10,25,50))
    if st.button("Analyze"):
        st.info("The analysis of randomly selected {} employees is shown below:".format(number1))
        random_customer()
    
elif st.checkbox("Top N Employees to Churn"):
    st.subheader("How many employees to select?")
    st.text("Please select the number of employees")
    number2=st.selectbox("", (5,10,25,50))
    if st.button("Show"):
        st.warning("Top {} customers to churn".format(number2))
        churn_customers()
    
elif st.checkbox("Top N Loyal Employees"):
    st.subheader("How many employees to select?")
    st.text("Please select the number of employees")
    number=st.selectbox("", (5,10,25,50))
    if st.button("Display"):
        st.success("Top {} loyal employees".format(number))
        loyal_customers()
            
            
