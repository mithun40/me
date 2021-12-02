font = ("Arial", 11)
import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
def read_table():
    sg.set_options(auto_size_buttons=True)
    layout = [[sg.Text('Dataset (a CSV file)', size=(16, 1)),sg.InputText(),
               sg.FileBrowse(file_types=(("CSV Files", "*.csv"),("Text Files", "*.txt")))],
               [sg.Submit(), sg.Cancel()]]

    window1 = sg.Window('Input file', layout)
    try:
        event, values = window1.read()
        window1.close()
    except:
        window1.close()
        return
    
    filename = values[0]
    
    if filename == '':
        return

    data = []
    header_list = []

    if filename is not None:
        fn = filename.split('/')[-1]
        try:                     
            if colnames_checked:
                df = pd.read_csv(filename, sep=',', engine='python')
                # Uses the first row (which should be column names) as columns names
                header_list = list(df.columns)
                # Drops the first row in the table (otherwise the header names and the first row will be the same)
                data = df[1:].values.tolist()
            else:
                df = pd.read_csv(filename, sep=',', engine='python', header=None)
                # Creates columns names for each column ('column0', 'column1', etc)
                header_list = ['column' + str(x) for x in range(len(df.iloc[0]))]
                df.columns = header_list
                # read everything else into a list of rows
                data = df.values.tolist()
            # NaN drop?
            if dropnan_checked:
                df = df.dropna()
                data = df.values.tolist()
            window1.close()
            return (df,data, header_list,fn)
        except:
            sg.popup_error('Error reading file')
            window1.close()
            return

def show_table(data, header_list, fn):    
    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  font=font,
                  pad=(25,25),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))]
    ]

    window = sg.Window(fn, layout, grab_anywhere=False)
    event, values = window.read()
    window.close()

def CI_cal(feaam, df):
  import scipy.stats as stats
  from numpy import sqrt
  sample_size=df.shape[0]
  sample_mean = df[feaam].mean()
  std_error = df[feaam].std()/sqrt(sample_size)
  CI=stats.norm.interval(0.95, loc=sample_mean, scale=std_error)
  return CI

def show_stats(df):
    stats = df.describe().T
    header_list = list(stats.columns)
    data = stats.values.tolist()
    CI_data=[]
    for i,d in enumerate(data):
        d.insert(0,list(stats.index)[i])
        k=stats.index[i]
        ci=CI_cal(k, df)
        CI_data.append(list(ci))
    header_list=['Feature']+header_list
    for i,d in enumerate(CI_data):
        d.insert(0,list(stats.index)[i])
    header_for_ci=['Minimum CI', 'Maximum CI']
    header_for_ci=['feature']+header_for_ci
    import scipy.stats as stats
    F, p=stats.f_oneway(df['r'], df['b1'], df['e'])
    # val=[F,p]
    # print(type(val))
    # print(type(F))
    # print(p)
    # heading=['F statistic', 'p value']
    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  font=font,
                  # pad=(10,10),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))],
 [sg.Text("95% confidence intervel", font=font)],
[sg.Table(values=CI_data, headings=header_for_ci, font=font, pad=(10,10), display_row_numbers=False, auto_size_columns=True, num_rows=min(25, len(data)))],
[sg.Text("One Way ANOVA", font=font)],
[sg.T(f'F statistic is {F} '), sg.T(f'p value is {p} ')],
# [sg.T(f'p value is {p} ')]

# [sg.Table(values=val, headings=heading, font=font)],

    ]

    window = sg.Window("Statistics", layout,grab_anywhere=False)
    event, values = window.read()
    window.close()

def sklearn_model(output_var):
    """
    Builds and fits a ML model
    """
    from sklearn.ensemble import RandomForestClassifier
    X = df.drop([output_var], axis=1)
    y = df[output_var]
    
    clf = RandomForestClassifier(max_depth=9, random_state=0)
    clf.fit(X, y)
    return clf, np.round(clf.score(X,y),3)
    #Incorporated New Model
def sklearn_model_DT(output_var):
    """
    Builds and fits a ML model
    """
    from sklearn.tree import DecisionTreeClassifier
    X = df.drop([output_var], axis=1)
    y = df[output_var]
    
    clf = DecisionTreeClassifier(max_leaf_nodes = 17, random_state = 0)
    clf.fit(X, y)
    return clf, np.round(clf.score(X,y),3)
def sklearn_model_GB(output_var):
    """
    Builds and fits a ML model
    """
    from sklearn.ensemble import GradientBoostingClassifier
    X = df.drop([output_var], axis=1)
    y = df[output_var]
    
    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=1.1, max_depth=6, random_state=0)
    clf.fit(X, y)
    return clf, np.round(clf.score(X,y),3)

def predict_model(a, b, e, clf):
  test=np.array([a,b,e])
  test=test.reshape(1,-1)
  y_prediction = clf.predict(test)
  return y_prediction
#=====================================================#
# Define the window's contents i.e. layout
layout = [
        [sg.Button('Load data',size=(10,1), enable_events=True, key='-READ-', font='Helvetica 16'),
        sg.Checkbox('Has column names?', size=(15,1), key='colnames-check',default=True),
        sg.Checkbox('Drop NaN entries?', size=(15,1), key='drop-nan',default=True)], 
         [sg.Button('Show data',size=(10,1),enable_events=True, key='-SHOW-', font='Helvetica 16',),
        sg.Button('Show stats',size=(15,1),enable_events=True, key='-STATS-', font='Helvetica 16',)],
            [sg.Text("", size=(50,1), key='-loaded-', pad=(5,5), font='Helvetica 14'),],
            #The below line is introduction of new 
    [sg.Text("Select the classifier", size=(18,2), pad=(5,5), font='Helvetica 12'),],
    [sg.Listbox(values=['Decision Tree', 'Random Forest', 'Gradient Boosting',], key='CLFR', size=(30, 6),enable_events=True),],
    [sg.Text("Select output column", size=(18,2), pad=(5,5), font='Helvetica 12'),],    
    [sg.Listbox(values=(''), key='colnames',size=(30,3),enable_events=True),],
    [sg.Text("", size=(50,1),key='-prediction-', pad=(5,5), font='Helvetica 12')], #
     [sg.ProgressBar(50, orientation='h', size=(100,20), key='progressbar')],
#This portion is used for taking input from users and predict the senerio 
       [sg.Text("Enter the data for predicting the fire", size=(100,2), justification='center', font='Helvetica 12')], 
        [sg.Text("Enter the value of r"), sg.Input(key='a',size=(20,3))],
        [sg.Text("Enter the value of b1"), sg.Input(key='b1',size=(20,3))],
        [sg.Text("Enter the value of e"), sg.Input(key='e',size=(20,3)),],
        [sg.Text("", size=(50,1), key='Predict_text', pad=(5,5), font='Helvetica 12')],
        [sg.ProgressBar(50, orientation='h', size=(100,20), key='progressbar1')],
        [sg.Button('Prediction',size=(10,1), enable_events=True, key='BRO', font='Helvetica 16')],     
        ]

# Create the window
window = sg.Window('Coal_India', layout, size=(900,600))
progress_bar = window['progressbar']
prediction_text = window['-prediction-']
#This two added for user input
progress_bar1 = window['progressbar1']
prediction_text1 = window['Predict_text']
colnames_checked = False
dropnan_checked = False
read_successful = False
# Event loop
while True:
  event, values = window.read()
  loaded_text = window['-loaded-']
  if event in (sg.WIN_CLOSED, 'Exit'):
    break
  if event == '-READ-':
    if values['colnames-check']==True:
      colnames_checked=True
    if values['drop-nan']==True:
      dropnan_checked=True
    try:
      df,data, header_list,fn = read_table()
      read_successful = True
    except:
      pass
    if read_successful:
      loaded_text.update("Datset loaded: '{}'".format(fn))
      col_vals = [i for i in df.columns]
      window.Element('colnames').Update(values=col_vals, )
  if event == '-SHOW-':
    if read_successful:
      show_table(data,header_list,fn)
    else:
      loaded_text.update("No dataset was loaded")
  if event=='BRO':
    prediction_text1.update("Predicting the given input...")
    for i in range(50):
      event, values = window.read(timeout=10)
      progress_bar1.UpdateBar(i + 1)
    score = predict_model(values['a'], values['b1'], values['e'], clf)
    if score==[0]:
      prediction_text1.update("No Fire...")
    else:
      prediction_text1.update("Alert people there might be fire!")
  if event=='-STATS-':
    if read_successful:
      show_stats(df)
    else:
      loaded_text.update("No dataset was loaded")

  if event=='colnames':
    if len(values['colnames'])!=0:
      output_var = values['colnames'][0]
      if output_var!='Label':
        sg.Popup("Wrong output column selected!", title='Wrong',font="Helvetica 14")
      else:
        prediction_text.update("Fitting model...")
        for i in range(50):
          event, values = window.read(timeout=10)
          progress_bar.UpdateBar(i + 1)
        if values['CLFR']==['Random Forest']:
          clf,score = sklearn_model(output_var)
          prediction_text.update("Accuracy of Random Forest model is: {}".format(score))
        elif values['CLFR']==['Decision Tree']:
          clf,score = sklearn_model_DT(output_var)
          prediction_text.update("Accuracy of Decision Tree model is: {}".format(score))
        elif values['CLFR']==['Gradient Boosting']:
          clf,score = sklearn_model_GB(output_var)
          prediction_text.update("Accuracy of Gradient Boosting model is: {}".format(score))
        else:
          sg.Popup("Please, select classifier!", title='Wrong',font="Helvetica 14")
  












########The source code found here#####

# font = ("Arial", 11)
# import PySimpleGUI as sg
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# def read_table():
#     sg.set_options(auto_size_buttons=True)
#     layout = [[sg.Text('Dataset (a CSV file)', size=(16, 1)),sg.InputText(),
#                sg.FileBrowse(file_types=(("CSV Files", "*.csv"),("Text Files", "*.txt")))],
#                [sg.Submit(), sg.Cancel()]]

#     window1 = sg.Window('Input file', layout)
#     try:
#         event, values = window1.read()
#         window1.close()
#     except:
#         window1.close()
#         return
    
#     filename = values[0]
    
#     if filename == '':
#         return

#     data = []
#     header_list = []

#     if filename is not None:
#         fn = filename.split('/')[-1]
#         try:                     
#             if colnames_checked:
#                 df = pd.read_csv(filename, sep=',', engine='python')
#                 # Uses the first row (which should be column names) as columns names
#                 header_list = list(df.columns)
#                 # Drops the first row in the table (otherwise the header names and the first row will be the same)
#                 data = df[1:].values.tolist()
#             else:
#                 df = pd.read_csv(filename, sep=',', engine='python', header=None)
#                 # Creates columns names for each column ('column0', 'column1', etc)
#                 header_list = ['column' + str(x) for x in range(len(df.iloc[0]))]
#                 df.columns = header_list
#                 # read everything else into a list of rows
#                 data = df.values.tolist()
#             # NaN drop?
#             if dropnan_checked:
#                 df = df.dropna()
#                 data = df.values.tolist()
#             window1.close()
#             return (df,data, header_list,fn)
#         except:
#             sg.popup_error('Error reading file')
#             window1.close()
#             return

# def show_table(data, header_list, fn):    
#     layout = [
#         [sg.Table(values=data,
#                   headings=header_list,
#                   font=font,
#                   pad=(25,25),
#                   display_row_numbers=False,
#                   auto_size_columns=True,
#                   num_rows=min(25, len(data)))]
#     ]

#     window = sg.Window(fn, layout, grab_anywhere=False)
#     event, values = window.read()
#     window.close()
# #########original show_stats#############
# # def show_stats(df):
# #     stats = df.describe().T
# #     # print(stats)
# #     header_list = list(stats.columns)
# #     data = stats.values.tolist()
# #     # print(data)
# #     # print(header_list)
# #     for i,d in enumerate(data):
# #         d.insert(0,list(stats.index)[i])
# #     header_list=['Feature']+header_list
# #     # print(header_list)
# #     layout = [
# #         [sg.Table(values=data,
# #                   headings=header_list,
# #                   font='Helvetica',
# #                   pad=(10,10),
# #                   display_row_numbers=False,
# #                   auto_size_columns=True,
# #                   num_rows=min(25, len(data)))]
# #     ]

# #     window = sg.Window("Statistics", layout, grab_anywhere=False)
# #     event, values = window.read()
# #     window.close()
# ####### MOdified the show stats for presenting input######
# def CI_cal(feaam, df):
#   import scipy.stats as stats
#   from numpy import sqrt
#   sample_size=df.shape[0]
#   sample_mean = df[feaam].mean()
#   std_error = df[feaam].std()/sqrt(sample_size)
#   CI=stats.norm.interval(0.95, loc=sample_mean, scale=std_error)
#   # print(format(CI, ".3f"))
#   # print("{0:.3f}".format(CI))
#   return CI

# def show_stats(df):
#     stats = df.describe().T
#     header_list = list(stats.columns)
#     data = stats.values.tolist()
#     # print(data)
#     # print(header_list)
#     CI_data=[]
#     for i,d in enumerate(data):
#         d.insert(0,list(stats.index)[i])
#         # print(type(d))
#         # print(d)
#         k=stats.index[i]
#         ci=CI_cal(k, df)
#         # print(type(ci))
#         CI_data.append(list(ci)) #This line for new table that is created
#         # CI_data.insert(0,list(stats.index)[i])  #CI data is inserted to new table 
#         # d.append(ci) # This line for first Table 
#     header_list=['Feature']+header_list
#     # print(type(CI_data))
#     # print(CI_data)
#     # new_head=['CI Min',]+['CI Max']
#     # header_list.append(new_head)
#     # header_list.append('MAX CI')
#     # header_list.append('Min CI')
#     for i,d in enumerate(CI_data):
#         d.insert(0,list(stats.index)[i])
#     header_for_ci=['Minimum CI', 'Maximum CI']
#     header_for_ci=['feature']+header_for_ci
#     # print()
#     # print(type(header_list))
#     # print(header_for_ci)
#     # print(CI_data)
#     # print(header_list)
#     # print(data)
#     layout = [
#         [sg.Table(values=data,
#                   headings=header_list,
#                   font=font,
#                   # pad=(10,10),
#                   display_row_numbers=False,
#                   auto_size_columns=True,
#                   num_rows=min(25, len(data)))],
# [sg.Table(values=CI_data, headings=header_for_ci, font=font, pad=(10,10), display_row_numbers=False, auto_size_columns=True, num_rows=min(25, len(data)))]
#     ]

#     window = sg.Window("Statistics", layout,grab_anywhere=False) #size=(900,900)) #grab_anywhere=False)
#     event, values = window.read()
#     window.close()


# def sklearn_model(output_var):
#     """
#     Builds and fits a ML model
#     """
#     from sklearn.ensemble import RandomForestClassifier
#     X = df.drop([output_var], axis=1)
#     y = df[output_var]
    
#     clf = RandomForestClassifier(max_depth=9, random_state=0)
#     clf.fit(X, y)
#     #print("Prediction accuracy {}".format(clf.score(X,y)))
#     return clf, np.round(clf.score(X,y),3)
#     #Incorporated New Model
# def sklearn_model_DT(output_var):
#     """
#     Builds and fits a ML model
#     """
#     # from sklearn.ensemble import RandomForestClassifier
#     from sklearn.tree import DecisionTreeClassifier
#     X = df.drop([output_var], axis=1)
#     y = df[output_var]
    
#     clf = DecisionTreeClassifier(max_leaf_nodes = 17, random_state = 0)
#     clf.fit(X, y)
#     #print("Prediction accuracy {}".format(clf.score(X,y)))
#     return clf, np.round(clf.score(X,y),3)
# def sklearn_model_GB(output_var):
#     """
#     Builds and fits a ML model
#     """
#     # from sklearn.ensemble import RandomForestClassifier
#     # from sklearn.tree import DecisionTreeClassifier
#     from sklearn.ensemble import GradientBoostingClassifier
#     X = df.drop([output_var], axis=1)
#     y = df[output_var]
    
#     clf = GradientBoostingClassifier(n_estimators=300, learning_rate=1.1, max_depth=6, random_state=0)
#     clf.fit(X, y)
#     #print("Prediction accuracy {}".format(clf.score(X,y)))
#     return clf, np.round(clf.score(X,y),3)

# def predict_model(a, b, e, clf):
#   # print(a)
#   # print(b)
#   # print(e)s
#   # print(clf)
#   test=np.array([a,b,e])
#   test=test.reshape(1,-1)
#   y_prediction = clf.predict(test)
#   return y_prediction
# # def print_fun(ME):
# #     print(ME)
# #=====================================================#
# # Define the window's contents i.e. layout
# layout = [
#         [sg.Button('Load data',size=(10,1), enable_events=True, key='-READ-', font='Helvetica 16'),
#         sg.Checkbox('Has column names?', size=(15,1), key='colnames-check',default=True),
#         sg.Checkbox('Drop NaN entries?', size=(15,1), key='drop-nan',default=True)], 
#          [sg.Button('Show data',size=(10,1),enable_events=True, key='-SHOW-', font='Helvetica 16',),
#         sg.Button('Show stats',size=(15,1),enable_events=True, key='-STATS-', font='Helvetica 16',)],
#             [sg.Text("", size=(50,1), key='-loaded-', pad=(5,5), font='Helvetica 14'),],
#             #The below line is introduction of new 
#     [sg.Text("Select the classifier", size=(18,2), pad=(5,5), font='Helvetica 12'),],
#     [sg.Listbox(values=['Decision Tree', 'Random Forest', 'Gradient Boosting',], key='CLFR', size=(30, 6),enable_events=True),],
#     [sg.Text("Select output column", size=(18,2), pad=(5,5), font='Helvetica 12'),],    
#     [sg.Listbox(values=(''), key='colnames',size=(30,3),enable_events=True),],
#     [sg.Text("", size=(50,1),key='-prediction-', pad=(5,5), font='Helvetica 12')], #
#      [sg.ProgressBar(50, orientation='h', size=(100,20), key='progressbar')],
# #This portion is used for taking input from users and predict the senerio 
#        [sg.Text("Enter the data for predicting the fire", size=(100,2), justification='center', font='Helvetica 12')], 
#         [sg.Text("Enter the value of r"), sg.Input(key='a',size=(20,3))],
#         [sg.Text("Enter the value of b1"), sg.Input(key='b1',size=(20,3))],
#         [sg.Text("Enter the value of e"), sg.Input(key='e',size=(20,3)),],
#         [sg.Text("", size=(50,1), key='Predict_text', pad=(5,5), font='Helvetica 12')],
#         [sg.ProgressBar(50, orientation='h', size=(100,20), key='progressbar1')],
#         [sg.Button('Prediction',size=(10,1), enable_events=True, key='BRO', font='Helvetica 16')],
#         # [sg.Button('Checking Pupose Only',size=(10,1),enable_events=True, key='-SHAW-', font='Helvetica 16')]      
#         ]

# # Create the window
# window = sg.Window('Coal_India', layout, size=(900,600))
# progress_bar = window['progressbar']
# prediction_text = window['-prediction-']
# #This two added for user input
# progress_bar1 = window['progressbar1']
# prediction_text1 = window['Predict_text']
# colnames_checked = False
# dropnan_checked = False
# read_successful = False
# # classifier_name=values['fac'][0]
# # Event loop
# while True:
#   event, values = window.read()
#   loaded_text = window['-loaded-']
#   if event in (sg.WIN_CLOSED, 'Exit'):
#     break
#   if event == '-READ-':
#     if values['colnames-check']==True:
#       colnames_checked=True
#     if values['drop-nan']==True:
#       dropnan_checked=True
#     try:
#       df,data, header_list,fn = read_table()
#       read_successful = True
#     except:
#       pass
#     if read_successful:
#       loaded_text.update("Datset loaded: '{}'".format(fn))
#       col_vals = [i for i in df.columns]
#       window.Element('colnames').Update(values=col_vals, )
#   if event == '-SHOW-':
#     if read_successful:
#       show_table(data,header_list,fn)
#     else:
#       loaded_text.update("No dataset was loaded")
# # #cHECKING
# #   if event=='-SHAW-':
# #       print_fun("Hello Comrade How are?")
# #cHECKING
#   if event=='BRO':
#     prediction_text1.update("Predicting the given input...")
#     for i in range(50):
#       event, values = window.read(timeout=10)
#       progress_bar1.UpdateBar(i + 1)
#     score = predict_model(values['a'], values['b1'], values['e'], clf)
#     if score==[0]:
#       prediction_text1.update("No Fire...")
#     else:
#       prediction_text1.update("Alert people there might be fire!")
#       # print(values['a'])
#       # if clf:
        
#       # else:
#       #   ssg.Popup("Wrong output column selected!", title='Wrong',font="Helvetica 14")

#     #   prediction_text1.update("Accuracy of Random Forest model is: {}".format(score))
#       #tILL tHIS POINTS
  
#   # if values['frac']=='Random Forest':
#   if event=='-STATS-':
#     if read_successful:
#       show_stats(df)
#     else:
#       loaded_text.update("No dataset was loaded")

#   # if event=='CLFR':
#   #   if len(values['CLFR'])!=0:
#   #     classifier_name=values['CLFR'][0]
#   #     if classifier_name=='Random Forest':
#   #       if event=='colnames':
#   #         if len(values['colnames'])!=0:
#   #           output_var = values['colnames'][0]
#   #           if output_var!='Label':
#   #             sg.Popup("Wrong output column selected!", title='Wrong',font="Helvetica 14")
#   #           else:
#   #             prediction_text.update("Fitting model...")
#   #             for i in range(50):
#   #               event, values = window.read(timeout=10)
#   #               progress_bar.UpdateBar(i + 1)
#   #             clf,score = sklearn_model(output_var)
#   #             prediction_text.update("Accuracy of Random Forest model is: {}".format(score))



#   # if classifier_name=='Random Forest':
#   #   if event=='colnames':
#   #     if len(values['colnames'])!=0:
#   #       output_var = values['colnames'][0]
#   #       if output_var!='Label':
#   #         sg.Popup("Wrong output column selected!", title='Wrong',font="Helvetica 14")
#   #       else:
#   #         prediction_text.update("Fitting model...")
#   #         for i in range(50):
#   #           event, values = window.read(timeout=10)
#   #           progress_bar.UpdateBar(i + 1)
#   #         clf,score = sklearn_model(output_var)
#   #         prediction_text.update("Accuracy of Random Forest model is: {}".format(score))



#   if event=='colnames':
#     if len(values['colnames'])!=0:
#       output_var = values['colnames'][0]
#       if output_var!='Label':
#         sg.Popup("Wrong output column selected!", title='Wrong',font="Helvetica 14")
#       else:
#         prediction_text.update("Fitting model...")
#         for i in range(50):
#           event, values = window.read(timeout=10)
#           progress_bar.UpdateBar(i + 1)
#         if values['CLFR']==['Random Forest']:
#           clf,score = sklearn_model(output_var)
#           prediction_text.update("Accuracy of Random Forest model is: {}".format(score))
#         elif values['CLFR']==['Decision Tree']:
#           clf,score = sklearn_model_DT(output_var)
#           prediction_text.update("Accuracy of Decision Tree model is: {}".format(score))
#         elif values['CLFR']==['Gradient Boosting']:
#           clf,score = sklearn_model_GB(output_var)
#           prediction_text.update("Accuracy of Gradient Boosting model is: {}".format(score))
#         else:
#           sg.Popup("Please, select classifier!", title='Wrong',font="Helvetica 14")

#           # print("Select Classifier")
#         # print(classifier_name)
# #FOR MANUAL INPUT
# #   if event=='-predicts-':
# #     print("Hello Comrade")
# #     prediction_text1.update("Predicting the given input...")
# #     for i in range(50):
# #       event, values = window.read(timeout=10)
# #       progress_bar1.UpdateBar(i + 1)
# #     score = predict_model(values[0], values[1], values[2])
# #     prediction_text1.update("Accuracy of Random Forest model is: {}".format(score))
  
