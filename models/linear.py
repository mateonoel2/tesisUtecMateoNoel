from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import abs, asc, col
import sys
from pyspark.ml import Pipeline

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        # Create a SparkSession
        spark = SparkSession.builder \
        .appName("PySpark Linear Regression")\
        .config("spark.executor.instances", "10")\
        .config("spark.executor.cores", "8")\
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()
        
        #total_count_data = 10000 
        print("89 días")

        # Load data from Parquet
        data = spark.read.parquet("../processed_datasets3/dataset").coalesce(1)\
                .orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).cache()
                #.filter((col("label") >= 0.291) & (col("label") <= 0.375))\
                #.limit(total_count_data)\
                #.filter((col("month") == 8) & (col("day_of_month") <= 3))\
        
        dataOG = data

        for a in range(2):
            if a == 0:
                #Short distances:
                data = dataOG.drop("first_time", "vehicle_id", "month", "day_of_month", "first_stop", "total_distance")
                print("Short distances:")
            elif a == 1:
                #Long distances:
                data = dataOG.drop("vehicle_id", "month", "day_of_month", "exit_time", "exit_stop", "distance")
                print("Long distances:")

            data.show(5)

            # Prepare data for regression training
            assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
            data = assembler.transform(data).select("features", "label") 

            # Calculate the split point
            #split_point = int(total_count_data * 0.9)
            split_point = int(data.count() * 0.9)

            # Split the data in order
            train_data = data.limit(split_point).cache() 
            test_data = data.subtract(train_data).cache()  

            #choose regressor
            for i in range(3):
                if i == 0:
                    regressor = GBTRegressor(featuresCol="features", labelCol="label")
                    print("GBTRegressor")
                elif i == 1:
                    regressor = RandomForestRegressor(featuresCol="features", labelCol="label")
                    print("RandomForestRegressor")
                elif i == 2:
                    regressor = LinearRegression(featuresCol="features", labelCol="label")
                    print("LinearRegression")

                # Train the regression model
                model = regressor.fit(train_data)

                # Make predictions on the test data
                predictions = model.transform(test_data)

                # Add a new column with the absolute difference between label and prediction
                predictions = predictions.withColumn("abs_diff", abs(predictions["label"] - predictions["prediction"]))

                total_count = predictions.count()

                # Calculate the percentage of data within a ranges of 10, 5, 1, and 0.5 minutes
                ranges = [(10*60)/86400, (5*60)/86400, 60/86400, 30/86400]

                # Iterate over the ranges and calculate the percentage of data within each range
                for r in ranges:
                    within_range_count = predictions.filter(predictions["abs_diff"] <= r).count()
                    percent_within_range = within_range_count / total_count * 100
                    print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r * 86400))

                # Evaluate the model using mean squared error and r2
                evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
                rmse = evaluator.evaluate(predictions)
                print("Root Mean Squared Error (RMSE) = {:.4f}".format(rmse))

                evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
                r2 = evaluator.evaluate(predictions)
                print("R-squared (R2) = {:.4f}\n".format(r2))

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()