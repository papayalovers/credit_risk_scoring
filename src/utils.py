import pandas as pd
import yaml

class ConfigManager:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = None

    def load_config(self) -> dict:
        """ 
        Load configuration parameters from a YAML file.

        Returns:
            dict: Configuration parameters.
        """
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        return self.config

    def update_config(self, key: str, value: any) -> None:
        """
        Update a configuration parameter and save it back to the YAML file.
        Args:
            key (str): The configuration key to update (can be nested using dot notation).
            value (any): The new value for the configuration key.
        Returns:
            None
        """
        if self.config is None:
            raise RuntimeError("Config not loaded. Please load the config before updating.")

        # split if the key is nested
        keys = key.split(".")
        cfg = self.config

        for k in keys[:-1]:
            # auto create dict if key doesn't exist
            if k not in cfg or not isinstance(cfg[k], dict):
                cfg[k] = {}  
            cfg = cfg[k]

        cfg[keys[-1]] = value

        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config, file)

        print(f"Updated: {key} = {value}")


class DataLoaderAndValidator:
    def __init__(self, config: dict) -> None:
        self.data_path = config.get('path', {}).get('raw_data', '')
        self.schema = config.get('schema', {})
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """ 
        Load data from a CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        pd.set_option('display.max_colwidth', None)
        self.data = pd.read_csv(self.data_path)


        # Create missing report of data columns
        missing_report = self.data.isna().sum().to_frame(name='missing_count').reset_index()
        missing_report['missing_percentage'] = ((missing_report['missing_count'] / len(self.data)) * 100).round(2).astype(str) + '%'
        # Create data types report
        result = list()
        for col in self.data.columns:
            result.append([col, self.data[col].dtypes, self.data[col].nunique(), self.data[col].unique()])
        data_types_report = pd.DataFrame(result, columns=['column_name', 'data_type', 'unique_count', 'unique_values'])
        # Merge reports
        data_reports = data_types_report.merge(missing_report, left_on='column_name', right_on='index').drop(columns=['index'], axis=1)

        print(f"Duplicated rows: {self.data.duplicated().sum()}")
        print(f"Data shape: {self.data.shape}")
        print()
        print("Data Overview:")
        print(50*"-")
        display(
            self.data.head(),
            data_reports,
            self.data.describe(include='object'),
            self.data.describe(),
        )

        return self.data
    
    def validate_data(self) -> bool:
        """ 
        Validate data against the schema.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        TYPE_MAPPING = {
            'integer': ['int64', 'int32'],
            'decimal': ['float64', 'float32'],
            'string': ['object', 'string'],
        }

        # Check if data is not loaded
        if self.data is None:
            raise RuntimeError("Data not loaded. Please load the data before validation.")
        
        # Store validation errors
        errors = list()

        # Check each colum from schema and dataset
        schema_cols = set(self.schema.keys())
        data_cols = set(self.data.columns)

        missing_in_data = schema_cols - data_cols # columns in schema but not in data
        extra_in_data = data_cols - schema_cols # columns in data but not in schema

        if missing_in_data:
            errors.append(f"Columns in schema but missing in data: {missing_in_data}")
        if extra_in_data:
            errors.append(f"Columns in data but not in schema: {extra_in_data}")
        
        for column, column_schema in self.schema.items():
            if column not in self.data.columns:
                continue # already reported as missing

            # Check data type
            data_type = self.data[column].dtype
            expected_type = column_schema.get('type', None)
            if expected_type in TYPE_MAPPING:
                if data_type not in TYPE_MAPPING[expected_type]:
                    errors.append(f"Column '{column}' has type '{data_type}', expected '{expected_type}'")
            else:
                errors.append(f"Column '{column}' has unexpected type '{data_type}'")

            # Check minimum value
            if 'minimum' in column_schema:
                min_value = column_schema['minimum']
                if not self.data[column].dropna().ge(min_value).all():
                    errors.append(f"Column '{column}' has values below minimum of {min_value}")
            
            # Check maximum value
            if 'maximum' in column_schema:
                max_value = column_schema['maximum']
                if not self.data[column].dropna().le(max_value).all():
                    errors.append(f"Column '{column}' has values above maximum of {max_value}")
            
            # Check enum values
            if 'enum' in column_schema:
                enum_values = set(column_schema['enum'])
                data_values = set(self.data[column].dropna().unique())
                invalid_values = data_values - enum_values
                if invalid_values:
                    errors.append(f"Column '{column}' has invalid enum values: {invalid_values}")

        if errors:
            print("Validation errors found:")
            for error in errors:
                print(f" - {error}")
            return 
        else:
            print("Data validation passed.")
            return  
        
class DataPreparation:
    def __init__(self, data: pd.DataFrame, config: dict)-> None:
        self.data = data
        self.features = config.get('features', {})
        self.target = config.get('target', [])
        self.path_processed_data = config.get('path', {}).get('processed_data', '')
    
    def split_input_output(self) -> tuple[pd.DataFrame, pd.Series]:
        """ 
        Split data into input features and target variable.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Input features and target variable.
        """
        features_columns = list() 

        for _, value_list in self.features.items():
            for col in value_list:
                if col not in self.data.columns:
                    raise ValueError(f"Feature column '{col}' not found in data.")
                features_columns.append(col)

        # ------ Print features data types
        print("Input Features Data Types:")
        print(f"Input Features Categorical: {self.features.get('categorical', [])}")
        print(f"Input Features Numerical: {self.features.get('numerical', [])}")
        print(100*"-")
        # ------ Extract input features and target variable
        X = self.data[features_columns]
        y = self.data[self.target]

        print(f"Input features shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        return X, y
    
    def split_train_test(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """ 
        Split data into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and testing sets.
        """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def serialized_data(self, data: any, name: str) -> None:
        """ 
        Serialize data to a file using joblib. 
        Args:
            data (any): Data to serialize.
            name (str): Name of the file to save the serialized data.
        Returns:
            None
        """
        import joblib
        file_path = f"{self.path_processed_data}/{name}.pkl"
        joblib.dump(data, file_path)
        print(f"Serialized data saved to {file_path}")

    def deserialize_data(self, name: str) -> any:
        """ 
        Deserialize data from a file using joblib.
        Args:
            name (str): Name of the file to load the serialized data from.
        Returns:
            any: Deserialized data.
        """
        import joblib
        file_path = f"{self.path_processed_data}/{name}.pkl"
        data = joblib.load(file_path)
        print(f"Deserialized data loaded from {file_path}")
        return data