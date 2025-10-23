import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as lambdaDocker from "aws-cdk-lib/aws-lambda";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as ssm from "aws-cdk-lib/aws-ssm";

export class BaselineStreamStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const modelGen   = ssm.StringParameter.valueForStringParameter(this, "/srag/MODEL_GEN");
    const retrieveUrl= ssm.StringParameter.valueForStringParameter(this, "/srag/RETRIEVE_URL");
    const topk       = ssm.StringParameter.valueForStringParameter(this, "/srag/STREAM_TOPK", 1) || "5";
    const maxtok     = ssm.StringParameter.valueForStringParameter(this, "/srag/STREAM_MAX_TOKENS", 1) || "80";

    const fn = new lambdaDocker.DockerImageFunction(this, "StreamHandler", {
      code: lambdaDocker.DockerImageCode.fromImageAsset("../services/baseline_stream"),
      memorySize: 512,
      timeout: cdk.Duration.seconds(60),
      tracing: lambda.Tracing.ACTIVE,
      environment: {
        MODEL_GEN: modelGen,
        RETRIEVE_URL: retrieveUrl,
        STREAM_TOPK: topk,
        STREAM_MAX_TOKENS: maxtok,
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ["bedrock:InvokeModelWithResponseStream", "bedrock:InvokeModel", "bedrock:Converse"],
      resources: ["*"],
    }));

    const furl = fn.addFunctionUrl({ authType: lambda.FunctionUrlAuthType.NONE });
    new cdk.CfnOutput(this, "FunctionURL", { value: furl.url });
  }
}
