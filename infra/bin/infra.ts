import * as cdk from "aws-cdk-lib";
import { BaselineSStack } from "../lib/baseline_s_stack";
import { BaselineStreamStack } from "../lib/baseline_stream_stack";

const app = new cdk.App();
new BaselineSStack(app, "BaselineSStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || "ap-northeast-2",
  },

});
new BaselineStreamStack(app, "BaselineStreamStack", {
  env: { 
    account: process.env.CDK_DEFAULT_ACCOUNT, 
    region: process.env.CDK_DEFAULT_REGION || "ap-northeast-2" 
  }
  
});
