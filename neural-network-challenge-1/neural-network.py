
        # Imports
        import pandas as pd
        import tensorflow as tf
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report
        from pathlib import Path
      
          
        # Read the csv into a Pandas DataFrame
        file_path = \"https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv\"\n",
        loans_df = pd.read_csv(file_path)
        
        #Review the DataFrame
        loans_df.head()
      
    
        # Review the data types associated with the columns\n",
        loans_df.dtypes
     
        # Check the credit_ranking value counts\n",
        loans_df[\"credit_ranking\"].value_counts()
     
        # Define the target set y using the credit_ranking column
        y = loans_df[\"credit_ranking\"]
        
        # Display a sample of y
        y[:5]
   
        # Define features set X by selecting all columns but credit_ranking
        X = loans_df.copy().drop(\"credit_ranking\", axis=1)
        
        # Review the features DataFrame
        X.head()
    
        # Split the preprocessed data into a training and testing dataset
        # Assign the function a random_state equal to 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.8)
      
        # Step 4: Use scikit-learn's `StandardScaler` to scale the features data."
      
        # Create a StandardScaler instance\n",
        scaler = StandardScaler().fit(X_train)
        
        # Fit the scaler to the features training dataset
        
        # Fit the scaler to the features training dataset
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = scaler.transform(X_train)
     
        ## Compile and Evaluate a Model Using a Neural Network
     
        ### Step 1: Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
      
        **Hint** You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.
     
        # Define the the number of inputs (features) to the model
        inputs = X_train_scaled.shape[1]
        
        # Review the number of features
        inputs
      
        # Define the number of hidden nodes for the first hidden layer
        l1_nodes = 8
        
        # Define the number of hidden nodes for the second hidden layer
        l2_nodes = 5
        
        # Define the number of neurons in the output layer
        number_output_neurons = 1
    
        # Create the Sequential model instance
        nn = Sequential()
       
        # Add the first hidden layer
        nn.add(Dense(units=l1_nodes, input_dim=inputs, activation=\"relu\"))
        
        # Add the second hidden layer
        nn.add(Dense(units=l2_nodes, activation=\"relu\"))
        
        # Add the output layer to the model specifying the number of output neurons and activation function
        nn.add(Dense(units=number_output_neurons, activation=\"sigmoid\"))
     
        # Display the Sequential model summary
        nn.summary()
   
        ### Step 2: Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.
    
        # Compile the Sequential model
        nn.compile(loss=\"binary_crossentropy\", optimizer=\"adam\", metrics=[\"accuracy\"])
     
        # Fit the model using 50 epochs and the training data
        model = nn.fit(X_train_scaled, y_train, epochs=50)
      
       ### Step 3: Evaluate the model using the test data to determine the model’s loss and accuracy.
      
        # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
        model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
        
        # Display the model loss and accuracy results
        print(f\"Loss: {model_loss}, Accuracy: {model_accuracy}\")
      
        ### Step 4: Save and export your model to a keras file, and name the file `student_loans.keras`.
      
        # Set the model's file path
        path = Path(\"saved_models/student_loan_model.keras\")
        
        # Export your model to a keras file
        nn.save(path)
      
        ## Predict Loan Repayment Success by Using your Neural Network Model
      
        ### Step 1: Reload your saved model.
    
        # Set the model's file path
        path = Path(\"saved_models/student_loan_model.keras\")
        
        # Load the model to a new object
        nn_model = tf.keras.models.load_model(path)
      
        ### Step 2: Make predictions on the testing data and save the predictions to a DataFrame.
      
        # Make predictions with the test data
        predictions = nn_model.predict(X_test_scaled)
       
        # Display a sample of the predictions
        predictions[:5]
      
        # Save the predictions to a DataFrame and round the predictions to binary results
        results = pd.DataFrame(y_test)
        results[\"predicted\"] = predictions.round()
        
        results
     
        ### Step 4: Display a classification report with the y test data and predictions
      
        # Print the classification report with the y test data and predictions
        print(classification_report(results[\"credit_ranking\"], results[\"predicted\"]))
      
        ## Discuss creating a recommendation system for student loans
      
        Briefly answer the following questions in the space provided:
        
        1. Describe the data that you would need to collect to build a recommendation system to recommend student loan options for students. Explain why this data would be relevant and appropriate.  \n",
        
        *As a creditor I would want to now a student's and their parent's, credit history, including: payment history, lines of credit, assets, and credit score, in addition to some more education focused metrics such as, choice of school and degree, graduation rates for that combination, their gpa, class rank, and average income for people who obtain that degree.*\n",
        
        2. Based on the data you chose to use in this recommendation system, would your model be using collaborative filtering, content-based filtering, or context-based filtering? Justify why the data you selected would be suitable for your choice of filtering method.\n",
        
        *Content Based: This is content based because you are training an outcome based on discrete metrics.*\n",
        
        3. Describe two real-world challenges that you would take into consideration while building a recommendation system for student loans. Explain why these challenges would be of concern for a student loan recommendation system.\n",
        
        *Some real world challenges with these metrics are that most people entering college don't have a credit history, using their parents credit scores can get around this if they are cosigning the loan but many students aren't receiving support from their parents even if they're well off, not to mention the fact that biasing off of low income families perpetuates cycles of poverty. By limiting resources based on degree type is problematic because it discourages people from entering into un-lucrative fields that are still important, such as nursing or the arts. Further more, the practice of keeping education and knowledge behind a paywall is abhorrent in the 21st century. Most countries that the US aligns it self with have free or cheep access to education while we force our young to start off their adult lives in crippling debt.*\n"
     
      
        **1. Describe the data that you would need to collect to build a recommendation system to recommend student loan options for students. Explain why this data would be relevant and appropriate.**
        
    
        **2. Based on the data you chose to use in this recommendation system, would your model be using collaborative filtering, content-based filtering, or context-based filtering? Justify why the data you selected would be suitable for your choice of filtering method.**\n",
        
        
        **3. Describe two real-world challenges that you would take into consideration while building a recommendation system for student loans. Explain why these challenges would be of concern for a student loan recommendation system.**\n",
       
        