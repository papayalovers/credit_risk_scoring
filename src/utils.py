import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import math 
import random 
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import numpy as np

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
        

    def load_data(self) -> pd.DataFrame:
        """ 
        Load data from a CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        pd.set_option('display.max_colwidth', None)
        df = pd.read_csv(self.data_path)
        return df

    def get_report(self, data: pd.DataFrame) -> None:
        '''
        Docstring for get_report
        
        Args:
            data (pd.DataFrame): The DataFrame to analyze.
        Returns:
            None: This function prints the report to the console.
        '''
        # Create missing report of data columns
        missing_report = data.isna().sum().to_frame(name='missing_count').reset_index()
        missing_report['missing_percentage'] = ((missing_report['missing_count'] / len(data)) * 100).round(2).astype(str) + '%'
        # Create data types report
        result = list()
        for col in data.columns:
            result.append([col, data[col].dtypes, data[col].nunique(), data[col].unique()])
        data_types_report = pd.DataFrame(result, columns=['column_name', 'data_type', 'unique_count', 'unique_values'])
        # Merge reports
        data_reports = data_types_report.merge(missing_report, left_on='column_name', right_on='index').drop(columns=['index'], axis=1)

        print(f"Duplicated rows: {data.duplicated().sum()}")
        print(f"Data shape: {data.shape}")
        print()
        print("Data Overview:")
        print(50*"-")
        display(
            data.head(),
            data_reports,
            data.describe(include='object'),
            data.describe(),
        )

    def validate_data(self, data: pd.DataFrame) -> bool:
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
        
        # Store validation errors
        errors = list()

        # Check each colum from schema and dataset
        schema_cols = set(self.schema.keys())
        data_cols = set(data.columns)

        missing_in_data = schema_cols - data_cols # columns in schema but not in data
        extra_in_data = data_cols - schema_cols # columns in data but not in schema

        if missing_in_data:
            errors.append(f"Columns in schema but missing in data: {missing_in_data}")
        if extra_in_data:
            errors.append(f"Columns in data but not in schema: {extra_in_data}")
        
        for column, column_schema in self.schema.items():
            if column not in data.columns:
                continue # already reported as missing

            # Check data type
            data_type = data[column].dtype
            expected_type = column_schema.get('type', None)
            if expected_type in TYPE_MAPPING:
                if data_type not in TYPE_MAPPING[expected_type]:
                    errors.append(f"Column '{column}' has type '{data_type}', expected '{expected_type}'")
            else:
                errors.append(f"Column '{column}' has unexpected type '{data_type}'")

            # Check minimum value
            if 'minimum' in column_schema:
                min_value = column_schema['minimum']
                if not data[column].dropna().ge(min_value).all():
                    errors.append(f"Column '{column}' has values below minimum of {min_value}")
            
            # Check maximum value
            if 'maximum' in column_schema:
                max_value = column_schema['maximum']
                if not data[column].dropna().le(max_value).all():
                    errors.append(f"Column '{column}' has values above maximum of {max_value}")
            
            # Check enum values
            if 'enum' in column_schema:
                enum_values = set(column_schema['enum'])
                data_values = set(data[column].dropna().unique())
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
    

class SerializationManager:
    def __init__(self, config: dict) -> None:
        self.path_processed_data = config.get('path', {}).get('processed_data', '')
        
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
    
class VisualizeData:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.features = config.get('features', {})
    
    def undisplay_axis(self, cols: list, axis: plt.Axes, fig: plt.Figure) -> None:
        """ 
        Hide the axis of a plot.

        Args:
            axis (plt.Axes): The axis to hide.
        Returns:
            None
        """

        for i in range(len(cols), len(axis)):
            fig.delaxes(axis[i])

    def plot_numeric_distributions(
        self, 
        data: pd.DataFrame,
        title: str,
        check_outliers: bool=False,
        vline_mean: bool=False,
        hue: str=''
    ) -> None:
        """ 
        Plot distribution of numeric features using seaborn.

        Args:
            data (pd.DataFrame): The DataFrame containing the data to plot.
            check_outliers (bool): Whether to check for outliers and plot them.
            title (str): Title of the plot.
            vline_mean (bool): Whether to add a vertical line for the mean value.
            hue (str): Column name for color encoding in the plot.
        Returns:
            None
        """
        sns.set_style("whitegrid")
        # len num features
        n_features = self.features.get('numerical', [])
        if not n_features:
            print("No numerical features found.")
            return
        
        num_cols = 2
        num_rows = math.ceil(len(n_features) / num_cols)
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(14, 10))
        ax = ax.flatten() if len(n_features) > 1 else [ax]

        cols = data.describe().columns.drop(['loan_status']) if 'loan_status' in data.describe().columns else data.describe().columns
        for subax, col in zip(ax, cols):
            data_range = f"{data[col].min()} - {data[col].max()}"

            if hue:
                sns.kdeplot(data=data, x=col, hue=hue, fill=True, ax=subax)
            else:
                sns.kdeplot(data=data, x=col, fill=True, ax=subax)

            if check_outliers:
                # Tambah vline untuk bounds jika cek outlier
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                if lower_bound < 0:
                    lower_bound = 0
                subax.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
                subax.axvline(upper_bound, color='red', linestyle='--', label='Upper Bound')
            
            if vline_mean:
                mean_value = data[col].mean()
                subax.axvline(mean_value, color='green', linestyle='-', label='Mean')

                subax.text(
                    mean_value, 
                    subax.get_ylim()[1] * 0.9, 
                    f'Mean: {mean_value:.2f}', 
                    color='green', 
                    ha='center',
                    va='top',
                    fontsize=9,
                    fontweight='bold'
                )

            subax.set_title(f'{col} Distribution', fontweight='bold')
            subax.set_ylabel('')
            subax.set_xlabel('')
            if not hue:
                subax.legend([data_range], title='ranges')
        # Hapus subplot yang tidak terpakai
        self.undisplay_axis(cols=cols, axis=ax, fig=fig) 

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    def plot_categorical_distributions(
        self, 
        data: pd.DataFrame, 
        title: str,
        hue: str='',
        check_target: bool=False
    ) -> None:
        """ 
        Plot distribution of categorical features using seaborn.

        Args:
            data (pd.DataFrame): The DataFrame containing the data to plot.
            title (str): Title of the plot.
            hue (str): Column name for color encoding in the plot.
        Returns:
            None
        """
        sns.set_style("whitegrid")
        cat_cols = self.features.get('categorical', [])
        if not cat_cols:
            print("No categorical features found.")
            return
        
        if check_target:
            fig, ax = plt.subplots(figsize=(8, 6))
            color = random.choice(sns.color_palette(palette='BrBG'))
            target_cols = self.config.get('target', [])
            data_plot = sns.countplot(data=data, x=target_cols[0], ax=ax, color=color)
            for container in data_plot.containers:
                data_plot.bar_label(container)
        
        else:
            num_cols = 2
            num_rows = math.ceil(len(cat_cols) / num_cols)
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 8))
            ax = ax.flatten() if len(cat_cols) > 1 else [ax]
            for subax, col in zip(ax.flatten(), cat_cols):
                if hue:
                    data_plot = sns.countplot(data=data, y=col, hue=hue, ax=subax, palette='husl')
                    
                    for container in data_plot.containers:
                        data_plot.bar_label(container)
                else:
                    color = random.choice(sns.color_palette(palette='BrBG'))
                    data_plot =sns.countplot(data=data, y=col, ax=subax, color=color)

                    for container in data_plot.containers:
                        data_plot.bar_label(container)

                subax.set_title(f'{col} Distribution', fontweight='bold')
                subax.set_ylabel('')
                subax.set_xlabel('')
            # Hapus subplot yang tidak terpakai
            self.undisplay_axis(cols=cat_cols, axis=ax, fig=fig) 

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

class PreprocessingPipeline:
    def __init__(
        self, 
        config: dict,
        is_log_transform: bool=False, 
        is_robust: bool=False,
    ):
        # instance attribut
        self.is_log_transform = is_log_transform
        self.is_robust = is_robust
        self.config = config
        self.num_cols = self.config.get('features', {}).get('numerical', [])
        self.cat_cols = self.config.get('features', {}).get('categorical', [])
        self.ohe = None 
        self.robust_scaler = None

    # Method Save and Load Object
    def save_pipeline(self, filepath='models') -> None:
        """ 
        Save an object to a file using joblib.

        Args:
            file_path (str): The path to the file where the object will be saved.
        Returns:
            None
        """
        import joblib
        if self.is_log_transform and self.is_robust:
            name = 'preprocessing_pipeline_log_robust'
        elif self.is_log_transform:
            name = 'preprocessing_pipeline_log'
        elif self.is_robust:
            name = 'preprocessing_pipeline_robust'
        else:
            name = 'preprocessing_pipeline'
        # 
        _name = f"{filepath}/{name}.pkl"
        joblib.dump(self, _name)
        print(f"Pipeline saved to {_name}")

    # Method untuk mapping ordinal
    def _encode_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method ini hanya untuk mapping atribut kategorik ordinal
        ---
        Output: pd.DataFrame
        '''
        df = df.copy()
        # Mapping Loan Grade
        df['loan_grade'] = df['loan_grade'].map({
            'A': 1, 'B': 2, 'C': 3,
            'D': 4, 'E': 5, 'F': 6, 'G': 7
        })
        # Mapping History Default
        df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({
            'N': 0,
            'Y': 1
        })

        return df
    
    # Method untuk train
    def fit(self, df: pd.DataFrame) -> object:
        ''' 
        Method ini meliputi mapping data ordinal, fitting ohe object sklearn, dan
        fitting robust scaler pada kondisi opsional. Semua dilakukan pada data train
        saja dan object ohe maupun robust scaler disimpan di atribut instance.
        ---
        Output: Instances
        '''
        # Panggil method mapping ordinal
        df = self._encode_ordinal(df)
        # inisialisasi ohe dan simpan di instance
        self.ohe = OneHotEncoder(
            handle_unknown='ignore', 
            drop='first',
            sparse_output=False
        )
        # 
        self.ohe.fit(df[self.cat_cols])

        # kondisional jika menggunakan robust
        if self.is_robust:
            self.robust_scaler = RobustScaler()
            self.robust_scaler.fit(df[self.num_cols])
        
        return self
    
    # Method transform data
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Method ini digunakan untuk transformasi seluruh fitur menggunakan atribut
        instance yang telah disimpan pada proses fitting sebelumnya.
        Kondisi opsional adalah penggunaan log_transform dan robust_scaler
        ---
        Output: pd.DataFrame 
        '''
        df = self._encode_ordinal(df.copy())
        # apply ohe
        ohe_arr = self.ohe.transform(df[self.cat_cols])
        ohe_df = pd.DataFrame(
            ohe_arr,
            columns=self.ohe.get_feature_names_out(self.cat_cols),
            index=df.index
        )
        # drop dan satukan data hasil ohe
        df = df.drop(columns=self.cat_cols, axis=1)
        df = pd.concat([df, ohe_df], axis=1)

        # kondisional log_transform transformasi
        if self.is_log_transform:
            df[self.num_cols] = np.log1p(df[self.num_cols])

        # kondisional robust scaler transform 
        if self.is_robust:
            df[self.num_cols] = self.robust_scaler.transform(df[self.num_cols])

        return df

    # Method untuk proses fit + transform
    def fit_transform(self, df: pd.DataFrame) -> None:
        ''' 
        Method proses penggabungan antara fit dan transform
        '''
        return self.fit(df).transform(df)

