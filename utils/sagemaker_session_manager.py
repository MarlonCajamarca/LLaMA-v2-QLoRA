import sagemaker
import boto3

class sagemaker_session_manager(object):
    def __init__(self, sagemaker_session_bucket):
        self.sess = sagemaker.Session()
        
        # sagemaker session bucket -> used for uploading data, models and logs
        # sagemaker will automatically create this bucket if it not exists
        if sagemaker_session_bucket is None and sess is not None:
            # set to default bucket if a bucket name is not given
            sagemaker_session_bucket = self.sess.default_bucket()
        print(f"sagemaker_session_bucket: {sagemaker_session_bucket}")
        
        # Get segemaker role
        try:
            self.role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client('iam')
            self.role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
        
        # Instantiate sagemaker session
        self.sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
        
    def get_session_and_role(self):
        return self.sess, self.role