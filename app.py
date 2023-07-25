from flask import Flask, render_template, request
import pandas as pd
import csv
import os
import glob
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

model = pickle.load(open('Classification_task.pkl','rb'))

def principal(monthly_rate,amount,emi,month):
    for i in range(month):
        interest=monthly_rate*amount
        p=emi-interest
        amount-=p
    return amount

def prepay(dti,income):
  if(dti<40):
    p=income/2
  else:
    p=income*3/4
  return p

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods = ['GET','POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        data = []
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
        data = pd.DataFrame(data) 
        data.columns = data.iloc[0]
        data = data[1:]
        # column_headers = list(data.columns.values)
        # print("The Column Header :", column_headers)
        
        #1 data = data[data['EverDelinquent']==0]
    
        #2 data = data[data['DTI'] != 0]
        
        #3 data['Occupancy'] = data['Occupancy'].replace('O','0')
        # data['Occupancy'] = data['Occupancy'].replace('I','1')
        # data['Occupancy'] = data['Occupancy'].replace('S','2')
        

       #  'CreditScore', 'Units', 'Occupancy', 'Channel', 'PropertyType',
       # 'OrigLoanTerm', 'NumBorrowers', 'MonthsDelinquent', 'MonthsInRepayment'
        
        
        cs = data.loc[1,'CreditScore']
        un = data.loc[1,'Units']
        
        data['Occupancy'] = data['Occupancy'].replace('O','0')
        data['Occupancy'] = data['Occupancy'].replace('I','1')
        data['Occupancy'] = data['Occupancy'].replace('S','2')
        
        oc = data.loc[1,'Occupancy']
        
        data['Channel'] = data['Channel'].replace('B','0')
        data['Channel'] = data['Channel'].replace('C','1')
        data['Channel'] = data['Channel'].replace('R','2')
        data['Channel'] = data['Channel'].replace('T','3')
        
        ch = data.loc[1,'Channel']
        
        data['PropertyType'] = data['PropertyType'].replace('CO','0')
        data['PropertyType'] = data['PropertyType'].replace('CP','1')
        data['PropertyType'] = data['PropertyType'].replace('LH','2')
        data['PropertyType'] = data['PropertyType'].replace('MH','3')
        data['PropertyType'] = data['PropertyType'].replace('PU','4')
        data['PropertyType'] = data['PropertyType'].replace('SF','5')
        data['PropertyType'] = data['PropertyType'].replace('X','6')
        
        pt = data.loc[1,'PropertyType']
        olt = data.loc[1,'OrigLoanTerm']
        
        data['NumBorrowers'] = data['NumBorrowers'].replace('2','1')
        data['NumBorrowers'] = data['NumBorrowers'].replace('1','0')
        data['NumBorrowers'] = data['NumBorrowers'].replace('0','2')
        
        nb = data.loc[1,'NumBorrowers']
        md = data.loc[1,'MonthsDelinquent']
        mr = data.loc[1,'MonthsInRepayment']
        
        features = [cs, un, oc, ch, pt, olt, nb, md, mr]
       
        feat_list = np.array(features, dtype=object)
        feat_list = feat_list.reshape(1,9)
        print(feat_list)
        
        print(feat_list.shape)
        print('Ever deliquency Result:')
        print(model.predict(feat_list))
        prediction = model.predict(feat_list)
        
        p = float(data.loc[1,'OrigUPB'])
        r = float(data.loc[1,'OrigInterestRate'])/12/100
        n = float(data.loc[1,'OrigLoanTerm'])
        result_1 = p * r * (1 + r)**n
        result_2 = ((1 + r)**n)-1
        emi = result_1/result_2
        print("EMI:",emi)
        
        
        totalpayment = emi * n
        intrestamt = totalpayment - p
        
        DTI = float(data.loc[1,'DTI'])
        I = float(emi/DTI)
        monthlyincome = I
        
        monthlyrate = float(data.loc[1,'OrigInterestRate'])
        monthlyrate = monthlyrate/12
        
        curprincipal = np.vectorize(principal)(monthlyrate, p, emi, int(data.loc[1,'MonthsInRepayment']))
        
        print("Curprincipal", curprincipal)
        prepayment = np.vectorize(prepay)(DTI ,(monthlyincome * 24))
        prepayment = abs(prepayment - (emi * 24))
        
        data['Occupancy'] = data['Occupancy'].replace('O','0')
        data['Occupancy'] = data['Occupancy'].replace('I','1')
        data['Occupancy'] = data['Occupancy'].replace('S','2')
        
        
        print("Prepayment amt:")
        print(prepayment)
        
        
        
        #return render_template('output2.html',variable = prepayment)
        
        return render_template('output.html',prediction=prediction, variable = prepayment)
        #return render_template('data.html',data=data.to_html())
        
if __name__ == '__main__':
    app.run(debug = False)