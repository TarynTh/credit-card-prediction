# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn


from flask import Flask ,jsonify , request ,session,render_template ,redirect,render_template_string,make_response

app = Flask(__name__)


import pandas as pd
from scipy.special import expit
import joblib
import numpy as np
import xgboost
import os

PATH_IMG = os.path.join('static','src')
app.config['UPLOAD_FOLDER'] =  PATH_IMG

df = pd.read_csv('mean_and_sd.csv')
df = df.set_index('Unnamed: 0')
dict_df = df.to_dict()


@app.route('/' ,  methods=["POST","GET"])
def test1():
    error = None
    username = None
    password = None
    gender = None
    Card_cate = None
    value = None
    Customer_Age = None
    Dependent_count = None
    Months_on_book = None
    Total_Relationship_Count = None
    Months_Inactive_12_mon = None
    Contacts_Count_12_mon = None
    Credit_Limit = None
    Total_Revolving_Bal = None
    Avg_Open_To_Buy = None
    Total_Amt_Chng_Q4_Q1 = None
    Total_Trans_Amt = None
    Total_Trans_Ct = None
    Total_Ct_Chng_Q4_Q1 = None
    Avg_Utilization_Ratio = None
    X = None

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'lightoff.png')


    a = ""
    b = ""
    offer = ""
    suggestion = ""
    suggestion2 = ""
    suggestion3 = ""
    
    
    try :

        if request.form and request.method == 'POST':
            #username = request.form['input1']
            #password = request.form['input2']

            ####################################

            Cus_num = request.form['Cus_num']
            gender = request.form['gender']
            Cus_edu = request.form['Cus_edu']
            Marital_sta = request.form['Marital_sta']
            Inc_cate = request.form['Inc_cate']
            Card_cate = request.form['Card_cate']

            ####################################

            Customer_Age = request.form['Cus_age']
            Months_on_book = request.form['Month_ob']
            Total_Relationship_Count = request.form['Total_Relationship_Count']
            Months_Inactive_12_mon = request.form['Months_Inactive_12_mon']
            Contacts_Count_12_mon = request.form['Contacts_Count_12_mon']
            Credit_Limit = request.form['Credit_Limit']
            Total_Revolving_Bal = request.form['Total_Revolving_Bal']
            Avg_Open_To_Buy = request.form['Avg_Open_To_Buy']
            Total_Amt_Chng_Q4_Q1 = request.form['Total_Amt_Chng_Q4_Q1']
            Total_Trans_Amt = request.form['Total_Trans_Amt']
            Total_Trans_Ct = request.form['Total_Trans_Ct']
            Total_Ct_Chng_Q4_Q1 = request.form['Total_Ct_Chng_Q4_Q1']
            Avg_Utilization_Ratio = request.form['Avg_Utilization_Ratio']


            ####################################
            value = [Customer_Age,
                    Months_on_book,
                    Total_Relationship_Count,
                    Months_Inactive_12_mon,
                    Contacts_Count_12_mon,
                    Credit_Limit,
                    Total_Revolving_Bal,
                    Avg_Open_To_Buy,
                    Total_Amt_Chng_Q4_Q1,
                    Total_Trans_Amt,
                    Total_Trans_Ct,
                    Total_Ct_Chng_Q4_Q1,
                    Avg_Utilization_Ratio]

            ###################################

            data_dict = {'Customer_Age':Customer_Age,
            'Months_on_book':Months_on_book,
            'Total_Relationship_Count':Total_Relationship_Count,
            'Months_Inactive_12_mon':Months_Inactive_12_mon,
            'Contacts_Count_12_mon':Contacts_Count_12_mon,
            'Credit_Limit':Credit_Limit,
            'Total_Revolving_Bal':Total_Revolving_Bal,
            'Avg_Open_To_Buy':Avg_Open_To_Buy,
            'Total_Amt_Chng_Q4_Q1':Total_Amt_Chng_Q4_Q1,
            'Total_Trans_Amt':Total_Trans_Amt,
            'Total_Trans_Ct':Total_Trans_Ct,
            'Total_Ct_Chng_Q4_Q1':Total_Ct_Chng_Q4_Q1,
            'Avg_Utilization_Ratio':Avg_Utilization_Ratio}

            data_dict_normalize = data_dict

            for key in data_dict:
                data_dict = {'Customer_Age':Customer_Age,
                            'Months_on_book':Months_on_book,
                            'Total_Relationship_Count':Total_Relationship_Count,
                            'Months_Inactive_12_mon':Months_Inactive_12_mon,
                            'Contacts_Count_12_mon':Contacts_Count_12_mon,
                            'Credit_Limit':Credit_Limit,
                            'Total_Revolving_Bal':Total_Revolving_Bal,
                            'Avg_Open_To_Buy':Avg_Open_To_Buy,
                            'Total_Amt_Chng_Q4_Q1':Total_Amt_Chng_Q4_Q1,
                            'Total_Trans_Amt':Total_Trans_Amt,
                            'Total_Trans_Ct':Total_Trans_Ct,
                            'Total_Ct_Chng_Q4_Q1':Total_Ct_Chng_Q4_Q1,
                            'Avg_Utilization_Ratio':Avg_Utilization_Ratio}
                
                for km in dict_df['mean']:
                    if key == km:
                        mean = dict_df['mean'][km]            
                for ksd in dict_df['sd']:
                    if key == ksd:
                        sd = dict_df['sd'][ksd]
                for nor in data_dict:
                    if key == nor:
                        data_dict_normalize[nor] = (float(data_dict[nor])-float(mean))/float(sd)

            data_sig = pd.DataFrame(data_dict_normalize,index=[0])

            data_sig2 = expit(data_sig)

            data_sig3 = data_sig2.to_numpy()

            data_sig4 = pd.DataFrame(data_sig3,columns=data_sig2.columns)

            loaded_model = joblib.load('model5.json')

            result = loaded_model.predict(data_sig4)
            result_inproba = loaded_model.predict_proba(data_sig4)
            #result = 1
            #result = usemodel(value)
            #result,result_inproba = usemodel(value)
            ###################################

            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'lightoff.png')
            Customer_Age = int(Customer_Age)

            if result == 1:
                a = "This person is likely to move to another bank!!"
                b = ""

                full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'light.gif')
                offer = 'Offer promotion to customer by :'

                if Card_cate == 'platinum':
                    
                    suggestion  = 'Lower interest rate for loan'
                    suggestion2 = 'Privilege lounge'
                    suggestion3 = 'If using more than 7000 usd, will get round trip domestic airline ticket.'

                elif Card_cate == 'gold':
                    suggestion = 'Lower interest rate for loan'
                    suggestion3 = 'If using more than 7000 usd, will get round trip domestic airline ticket.'

                elif Card_cate == 'silver':
                    suggestion = 'Lower interest rate for loan'


                elif Card_cate == 'blue':

                    if Inc_cate == '80k-120k' or Inc_cate == 'mt120k':
                        suggestion = 'Encorage to upgrade their card by discount yearly fee by 50%.'

                    if  Customer_Age >= 20 and Customer_Age <= 40 :
                        suggestion2 = 'Gift voucher to make them repurchase.'

                    elif  Customer_Age >= 40 and Customer_Age <= 54 :
                        suggestion2 = 'Points multiplier 3x. The more you pay, the more you will gain.'
                        
                    elif  Customer_Age >= 55 :
                        suggestion2 = 'More cash back offer for specific goods.'

                else :
        
                    suggestion = Card_cate
                    

            else :
                a = ""
                b = 'This person is likely to continue using our bank services'  
                suggestion = 'Maintain good relationships with customer'
                full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'lightoff.png')

            ####################################

            print("username : ", username)
            print("password : ", password)

            if Credit_Limit == ''  :
                error = True 
                print("!!!!!")


        

        return render_template('test.html' , 
                                error = error ,

                                full_filename = full_filename ,
                                
                                show = True , 
                             
                                y_pred = a ,

                                y_pred_2 = b ,


                                offer = offer ,
                                suggestion = suggestion ,
                                suggestion2 = suggestion2 ,
                                suggestion3 = suggestion3 ,

                                
                               )

    except Exception as e:
        print("Error is : " , str(e))
        #return("Error is : " + str(e))
        return("กลับไปกรอกให้ครบดิเฮ้ย")

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0', threaded=True, port=6969)

# structure python
#     MVC module view controller  component
#     OOP object orented programing component []