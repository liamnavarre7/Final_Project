import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def clean_data(health_data_df):
    health_header = ["Active Energy (kcal)", "Apple Exercise Time (min)", "Apple Stand Hour (count)", "Apple Stand Time (min)", "Basal Body Temperature (degF)", "Basal Energy Burned (kcal)", "Blood Alcohol Content (%)", "Blood Glucose (mg/dL)", "Blood Oxygen Saturation (%)", "Blood Pressure [Systolic] (mmHg)", "Blood Pressure [Diastolic] (mmHg)", "Body Fat Percentage (%)", "Body Mass Index (count)", "Body Temperature (degF)", "Calcium (mg)", "Carbohydrates (g)", "Chloride (mg)", "Chromium (mcg)", "Copper (mg)", "Cycling Distance (mi)", "Dietary Biotin (mcg)", "Dietary Caffeine (mg)", "Dietary Cholesterol (mg)", "Dietary Energy (kcal)", "Dietary Sugar (g)", "Dietary Water (fl_oz_us)", "Distance Downhill Snow Sports (mi)", "Environmental Audio Exposure (dBASPL)", "Fiber (g)", "Flights Climbed (count)", "Folate (mcg)", "Forced Expiratory Volume 1 (L)", "Forced Vital Capacity (L)", "Handwashing (s)", "Headphone Audio Exposure (dBASPL)", "Heart Rate [Min] (count/min)", "Heart Rate [Max] (count/min)", "Heart Rate [Avg] (count/min)", "Heart Rate Variability (ms)", "Height (cm)", "High Heart Rate Notifications [Min] (count)", "High Heart Rate Notifications [Max] (count)", "High Heart Rate Notifications [Avg] (count)", "Inhaler Usage (count)", "Insulin Delivery (IU)", "Iodine (mcg)", "Iron (mg)", "Irregular Heart Rate Notifications [Min] (count)", "Irregular Heart Rate Notifications [Max] (count)", "Irregular Heart Rate Notifications [Avg] (count)", "Lean Body Mass (lb)", "Low Heart Rate Notifications [Min] (count)", "Low Heart Rate Notifications [Max] (count)", "Low Heart Rate Notifications [Avg] (count)", "Magnesium (mg)", "Manganese (mg)", "Mindful Minutes (min)", "Molybdenum (mcg)", "Monounsaturated Fat (g)", "Niacin (mg)", "Number of Times Fallen (count)", "Pantothenic Acid (mg)", "Peak Expiratory Flow Rate (L/min)", "Peripheral Perfusion Index (%)", "Polyunsaturated Fat (g)", "Potassium (mg)", "Protein (g)", "Push Count (count)", "Respiratory Rate (count/min)", "Resting Heart Rate (count/min)", "Riboflavin (mg)", "Saturated Fat (g)","Selenium (mcg)","Sexual Activity [Unspecified] (count)","Sexual Activity [Protection Used] (count)","Sexual Activity [Protection Not Used] (count)","Six-Minute Walking Test Distance (m)","Sleep Analysis [Asleep] (hr)","Sleep Analysis [In Bed] (hr)","Sodium (mg)","Stair Speed: Down (ft/s)","Stair Speed: Up (ft/s)","Step Count (count)","Swimming Distance (yd)","Swimming Stroke Count (count)","Thiamin (mg)","Toothbrushing (s)","Total Fat (g)","VO2 Max (ml/(kg·min))","Vitamin A (mcg)","Vitamin B12 (mcg)","Vitamin B6 (mg)","Vitamin C (mg)","Vitamin D (mcg)","Vitamin E (mg)","Vitamin K (mcg)","Waist Circumference (in)","Walking + Running Distance (mi)","Walking Asymmetry Percentage (%)","Walking Double Support Percentage (%)","Walking Heart Rate Average (count/min)","Walking Speed (mi/hr)","Walking Step Length (in)","Weight & Body Mass (lb)","Wheelchair Distance (mi)"]
    
    for element in health_header:
        if health_data_df[element].isnull().sum() / len(health_data_df) > 0.75: # for loop to get rid of columns with more than 75% of null values 
            health_data_df.drop(element, 1, inplace=True)

        
    health_data_df.drop("Zinc (mg) ", axis=1, inplace=True)
    health_data_df.drop("Walking Asymmetry Percentage (%)", axis=1, inplace=True)
    health_data_df.drop("Walking Speed (mi/hr)", axis=1, inplace=True)
    health_data_df.drop("Walking Step Length (in)", axis=1, inplace=True)
    
    
    health_data_df.interpolate("linear", inplace=True)
    health_data_df = health_data_df.bfill(axis=1, inplace=False, limit=None)
    health_data_df = health_data_df.ffill(axis=1, inplace=False, limit=None)


    liam_health_date_ser = health_data_df.index
    week_day_ser = []

    for element in liam_health_date_ser:
        date_obj = datetime.datetime.strptime(element, "%Y-%m-%d %H:%M:%S")
        week_day_ser.append(date_obj.weekday())

    health_data_df["Day of Week"] = week_day_ser
    return health_data_df

def load_test_value(test, data_frame):
    
    dum_list = []
    dum_list = pd.Series(dum_list)
    for i in range(len(data_frame)):
        dum_list[i] = test
        i += 1
    dum_list = list(dum_list)
    data_frame["Gonzaga"] = dum_list
    return data_frame

def steps_t_test(gonzaga_steps_2021, gonzaga_steps_2020, santa_steps_2021, santa_steps_2020):
    t_steps_liam, p_steps_liam = stats.ttest_rel(gonzaga_steps_2021,gonzaga_steps_2020)
    t_steps_ana, p_steps_ana = stats.ttest_rel(santa_steps_2021, santa_steps_2020)
    print("t-computed for Gonzaga steps:", t_steps_liam, "p-value for Gonzaga steps:", p_steps_liam)
    print("t-computed for Santa Clara steps:", t_steps_ana, "p-value for Santa Clara steps:", p_steps_ana)
    print()

def walk_dis_t_test(gonzaga_walk_2021, gonzaga_walk_2020, santa_walk_2021, santa_walk_2020):
    t_walk_dis_liam, p_walk_dis_liam = stats.ttest_rel(gonzaga_walk_2021, gonzaga_walk_2020)
    t_walk_dis_ana, p_walk_dis_ana = stats.ttest_rel(santa_walk_2021, santa_walk_2020)
    print("t-computed for Gonzaga walking distance:", t_walk_dis_liam, "p-value for Gonzaga walking distance:", p_walk_dis_liam)
    print("t-computed for Santa Clara walking distance:", t_walk_dis_ana,"p-value for Santa Clara walking distance:", p_walk_dis_ana)
    print()

def flights_t_test(gonzaga_flights_2021, gonzaga_flights_2020, santa_flights_2021, santa_flights_2020):
    t_flights_liam, p_flights_liam = stats.ttest_rel(gonzaga_flights_2021, gonzaga_flights_2020)
    t_flights_ana, p_flights_ana = stats.ttest_rel(santa_flights_2021, santa_flights_2020)
    print("t-computed for Gonzaga flights climbed:", t_flights_liam, "p-value for Gonzaga flights climbed:", p_flights_liam)
    print("t-computed for Santa Clara flights climbed:", t_flights_ana,"p-value for Santa Clara flights climbed:", p_flights_ana)

def scatter_steps(gonzaga_steps_2020, gonzaga_steps_2021, santa_steps_2020, santa_steps_2021):
    # For Steps
    plt.figure(figsize=(8,6), dpi=80)
    x = [0,12000]
    y = [0,17500]

    plt.scatter(gonzaga_steps_2020, gonzaga_steps_2021, s= 100, c = "blue", marker = ("."), label="Gonzaga (N=" + str(len(gonzaga_steps_2021)) +")")
    plt.scatter(santa_steps_2020, santa_steps_2021, s= 100, c = "red", marker = ("+"), label="Santa Clara (N=" +str(len(santa_steps_2021)) +")")
    plt.plot(x, y, color = "black", linestyle = "dashed", linewidth = 2.0, label = "No Change")
    plt.title("Steps (N=" +str(len(gonzaga_steps_2021) + len(santa_steps_2021)) + ")")
    plt.xlabel("Fall 2020 Step Total")
    plt.ylabel("Fall 2021 Step Total")
    plt.legend(loc = 4)
    plt.show()

def scatter_dis(gonzaga_dis_2020, gonzaga_dis_2021, santa_dis_2020, santa_dis_2021):
    plt.figure(figsize=(8,6), dpi=80)
    x = [0,7]
    y = [0,9]

    plt.scatter(gonzaga_dis_2020, gonzaga_dis_2021, s= 100, c = "blue", marker = ("."), label="Gonzaga (N=" + str(len(gonzaga_dis_2021)) +")")
    plt.scatter(santa_dis_2020, santa_dis_2021, s= 100, c = "red", marker = ("+"), label="Santa Clara (N=" +str(len(santa_dis_2021)) +")")
    plt.plot(x, y, color = "black", linestyle = "dashed", linewidth = 2.0, label = "No Change")
    plt.title("Walking Distance(mi) (N=" +str(len(gonzaga_dis_2021) + len(santa_dis_2021)) + ")")
    plt.xlabel("Fall 2020 Walking Distance Total")
    plt.ylabel("Fall 2021 Walking Distance Total")
    plt.legend(loc = 4)
    plt.show()

def scatter_flights(gonzaga_flights_2020, gonzaga_flights_2021, santa_flights_2020, santa_flights_2021):
    plt.figure(figsize=(8,6), dpi=80)
    x = [0,25]
    y = [0,60]

    plt.scatter(gonzaga_flights_2020, gonzaga_flights_2021, s= 100, c = "blue", marker = ("."), label="Gonzaga (N=" + str(len(gonzaga_flights_2020)) +")")
    plt.scatter(santa_flights_2020, santa_flights_2021, s= 100, c = "red", marker = ("+"), label="Santa Clara (N=" +str(len(santa_flights_2021)) +")")
    plt.plot(x, y, color = "black", linestyle = "dashed", linewidth = 2.0, label = "No Change")
    plt.title("Flights Climbed (N=" +str(len(gonzaga_flights_2020) + len(santa_flights_2021)) + ")")
    plt.xlabel("Fall 2020 Flights of Stairs Climbed Total")
    plt.ylabel("Fall 2021 Flights of Stairs Climbed Total")
    plt.legend(loc = 1)
    plt.show()

def bar_steps(gonzaga_steps_2020, gonzaga_steps_2021, santa_steps_2020, santa_steps_2021):
    plt.figure(figsize=(8, 6), dpi=80) 
    plt.bar("Gonzaga Step Average, Fall 2021", gonzaga_steps_2021.mean())
    plt.bar("Gonzaga Step Average, Fall 2020", gonzaga_steps_2020.mean())
    plt.bar("Santa Clara Step Average, Fall 2021", santa_steps_2021.mean())
    plt.bar("Santa Clara Step Average, Fall 2020", santa_steps_2020.mean())
    plt.title("Average Amount of Steps Per Day for Fall of 2020 and Fall of 2021")
    plt.ylabel("Average Step Count Per Day")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def bar_dis(gonzaga_dis_2020, gonzaga_dis_2021, santa_dis_2020, santa_dis_2021):
    plt.figure(figsize=(8, 6), dpi=80) 
    plt.bar("Gonzaga Walking Distance Average, Fall 2021(Mile)", gonzaga_dis_2021.mean())
    plt.bar("Gonzaga Walking Distance Average, Fall 2020(Mile)", gonzaga_dis_2020.mean())
    plt.bar("Santa Clara Walking Distance Average, Fall 2021(Mile)", santa_dis_2021.mean())
    plt.bar("Santa Clara Walking Distance Average, Fall 2020(Mile)", santa_dis_2020.mean())
    plt.title("Average Distance Walked Per Day for Fall of 2020 and Fall of 2021")
    plt.ylabel("Average Distance Walked Per Day(Mile)")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def bar_flights(gonzaga_flights_2020, gonzaga_flights_2021, santa_flights_2020, santa_flights_2021):
    plt.figure(figsize=(8, 6), dpi=80) 
    plt.bar("Gonzaga Flights of Stairs Climbed Average, Fall 2021", gonzaga_flights_2021.mean())
    plt.bar("Gonzaga Flights of Stairs Climbed Average, Fall 2020", gonzaga_flights_2020.mean())
    plt.bar("Santa Clara Flights of Stairs Climbed Average, Fall 2021", santa_flights_2021.mean())
    plt.bar("Santa Clara Flights of Stairs Climbed Average, Fall 2020", santa_flights_2020.mean())
    plt.title("Average Flights of Stairs Climber Per Day for Fall of 2020 and Fall of 2021")
    plt.ylabel("Average Flights of Stairs Climbed Per Day")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def steps_t_test_ind(gonzaga_steps_2021, gonzaga_steps_2020, santa_steps_2021, santa_steps_2020):
    t_steps_liam, p_steps_liam = stats.ttest_ind(gonzaga_steps_2021,gonzaga_steps_2020)
    t_steps_ana, p_steps_ana = stats.ttest_ind(santa_steps_2021, santa_steps_2020)
    print("t-computed for Gonzaga steps:", t_steps_liam, "p-value for Gonzaga steps:", p_steps_liam)
    print("t-computed for Santa Clara steps:", t_steps_ana, "p-value for Santa Clara steps:", p_steps_ana)
    print()

def walk_dis_t_test_ind(gonzaga_walk_2021, gonzaga_walk_2020, santa_walk_2021, santa_walk_2020):
    t_walk_dis_liam, p_walk_dis_liam = stats.ttest_ind(gonzaga_walk_2021, gonzaga_walk_2020)
    t_walk_dis_ana, p_walk_dis_ana = stats.ttest_ind(santa_walk_2021, santa_walk_2020)
    print("t-computed for Gonzaga walking distance:", t_walk_dis_liam, "p-value for Gonzaga walking distance:", p_walk_dis_liam)
    print("t-computed for Santa Clara walking distance:", t_walk_dis_ana,"p-value for Santa Clara walking distance:", p_walk_dis_ana)
    print()

def flights_t_test_ind(gonzaga_flights_2021, gonzaga_flights_2020, santa_flights_2021, santa_flights_2020):
    t_flights_liam, p_flights_liam = stats.ttest_ind(gonzaga_flights_2021, gonzaga_flights_2020)
    t_flights_ana, p_flights_ana = stats.ttest_ind(santa_flights_2021, santa_flights_2020)
    print("t-computed for Gonzaga flights climbed:", t_flights_liam, "p-value for Gonzaga flights climbed:", p_flights_liam)
    print("t-computed for Santa Clara flights climbed:", t_flights_ana,"p-value for Santa Clara flights climbed:", p_flights_ana)

def scatter_steps_ind(gonzaga_steps_2020, gonzaga_steps_2021, santa_steps_2020, santa_steps_2021):
    # For Steps
    plt.figure(figsize=(8,6), dpi=80)
    x = [0,12000]
    y = [0,17500]

    plt.scatter(gonzaga_steps_2020, gonzaga_steps_2021, s= 100, c = "blue", marker = ("."), label="Gonzaga (N=" + str(len(gonzaga_steps_2021)) +")")
    plt.scatter(santa_steps_2020, santa_steps_2021, s= 100, c = "red", marker = ("+"), label="Santa Clara (N=" +str(len(santa_steps_2021)) +")")
    plt.plot(x, y, color = "black", linestyle = "dashed", linewidth = 2.0, label = "No Change")
    plt.title("Steps (N=" +str(len(gonzaga_steps_2021) + len(santa_steps_2021)) + ")")
    plt.xlabel("Fall 2020 Friday Step Total")
    plt.ylabel("Fall 2021 Monday Step Total")
    plt.legend(loc = 4)
    plt.show()

def scatter_dis_ind(gonzaga_dis_2020, gonzaga_dis_2021, santa_dis_2020, santa_dis_2021):
    plt.figure(figsize=(8,6), dpi=80)
    x = [0,7]
    y = [0,9]

    plt.scatter(gonzaga_dis_2020, gonzaga_dis_2021, s= 100, c = "blue", marker = ("."), label="Gonzaga (N=" + str(len(gonzaga_dis_2021)) +")")
    plt.scatter(santa_dis_2020, santa_dis_2021, s= 100, c = "red", marker = ("+"), label="Santa Clara (N=" +str(len(santa_dis_2021)) +")")
    plt.plot(x, y, color = "black", linestyle = "dashed", linewidth = 2.0, label = "No Change")
    plt.title("Walking Distance(mi) (N=" +str(len(gonzaga_dis_2021) + len(santa_dis_2021)) + ")")
    plt.xlabel("Fall 2020 Friday Walking Distance Total")
    plt.ylabel("Fall 2021 Monday Walking Distance Total")
    plt.legend(loc = 4)
    plt.show()

def scatter_flights_ind(gonzaga_flights_2020, gonzaga_flights_2021, santa_flights_2020, santa_flights_2021):
    plt.figure(figsize=(8,6), dpi=80)
    x = [0,25]
    y = [0,60]

    plt.scatter(gonzaga_flights_2020, gonzaga_flights_2021, s= 100, c = "blue", marker = ("."), label="Gonzaga (N=" + str(len(gonzaga_flights_2020)) +")")
    plt.scatter(santa_flights_2020, santa_flights_2021, s= 100, c = "red", marker = ("+"), label="Santa Clara (N=" +str(len(santa_flights_2021)) +")")
    plt.plot(x, y, color = "black", linestyle = "dashed", linewidth = 2.0, label = "No Change")
    plt.title("Flights Climbed (N=" +str(len(gonzaga_flights_2020) + len(santa_flights_2021)) + ")")
    plt.xlabel("Fall 2020 Friday Flights of Stairs Climbed Total")
    plt.ylabel("Fall 2021 Monday Flights of Stairs Climbed Total")
    plt.legend(loc = 1)
    plt.show()

def bar_steps_ind(gonzaga_steps_2020, gonzaga_steps_2021, santa_steps_2020, santa_steps_2021):
    plt.figure(figsize=(8, 6), dpi=80) 
    plt.bar("Gonzaga Monday Step Average, Fall 2021", gonzaga_steps_2021.mean())
    plt.bar("Gonzaga Friday Step Average, Fall 2020", gonzaga_steps_2020.mean())
    plt.bar("Santa Monday Clara Step Average, Fall 2021", santa_steps_2021.mean())
    plt.bar("Santa Friday Clara Step Average, Fall 2020", santa_steps_2020.mean())
    plt.title("Average Amount of Steps Per Friday/Monday for Fall of 2020 and Fall of 2021")
    plt.ylabel("Average Step Count Per Day")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def bar_dis_ind(gonzaga_dis_2020, gonzaga_dis_2021, santa_dis_2020, santa_dis_2021):
    plt.figure(figsize=(8, 6), dpi=80) 
    plt.bar("Gonzaga Monday Walking Distance Average, Fall 2021(Mile)", gonzaga_dis_2021.mean())
    plt.bar("Gonzaga Friday Walking Distance Average, Fall 2020(Mile)", gonzaga_dis_2020.mean())
    plt.bar("Santa Clara Monday Walking Distance Average, Fall 2021(Mile)", santa_dis_2021.mean())
    plt.bar("Santa Clara Friday Walking Distance Average, Fall 2020(Mile)", santa_dis_2020.mean())
    plt.title("Average Distance Walked Per Friday/Monday for Fall of 2020 and Fall of 2021")
    plt.ylabel("Average Distance Walked Per Day(Mile)")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def bar_flights_ind(gonzaga_flights_2020, gonzaga_flights_2021, santa_flights_2020, santa_flights_2021):
    plt.figure(figsize=(8, 6), dpi=80) 
    plt.bar("Gonzaga Monday Flights of Stairs Climbed Average, Fall 2021", gonzaga_flights_2021.mean())
    plt.bar("Gonzaga Friday Flights of Stairs Climbed Average, Fall 2020", gonzaga_flights_2020.mean())
    plt.bar("Santa Clara Monday Flights of Stairs Climbed Average, Fall 2021", santa_flights_2021.mean())
    plt.bar("Santa Clara Friday Flights of Stairs Climbed Average, Fall 2020", santa_flights_2020.mean())
    plt.title("Average Flights of Stairs Climber Per Friday/Monday for Fall of 2020 and Fall of 2021")
    plt.ylabel("Average Flights of Stairs Climbed Per Day")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def hold_out_decision_tree_cross(X, Y):
    scalar = MinMaxScaler()
    X = scalar.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state= 0, stratify=Y)
    knn_clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

    knn_clf.fit(X_train, Y_train)
    Y_predicted = knn_clf.predict(X_test)
    #print(Y_predicted)
    accuracy = accuracy_score(Y_test, Y_predicted)
    print("Hold Out Method:", accuracy) 

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, Y_train)
    Y_predicted = tree_clf.predict(X_test)
    #print(Y_predicted)
    accuracy = accuracy_score(Y_test, Y_predicted)
    print("Decision Tree Accuracy:", accuracy)

    for clf in [knn_clf, tree_clf]: 
        # better approach 
        Y_predicted = cross_val_predict(clf, X, Y, cv=5)
        #print(Y_predicted)
        accuracy = accuracy_score(Y, Y_predicted)
        print("Cross Validation Accuracy:", accuracy)
