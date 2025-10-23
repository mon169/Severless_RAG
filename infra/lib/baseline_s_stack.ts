import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as lambdaDocker from "aws-cdk-lib/aws-lambda";
import * as apigw from "aws-cdk-lib/aws-apigateway";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as ssm from "aws-cdk-lib/aws-ssm";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ddb from "aws-cdk-lib/aws-dynamodb";

export class BaselineSStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ---------- SSM 파라미터 ----------
    const bucketName   = ssm.StringParameter.valueForStringParameter(this, "/srag/BUCKET_NAME");
    const lanceUri     = ssm.StringParameter.valueForStringParameter(this, "/srag/LANCE_URI");
    const lanceTable   = ssm.StringParameter.valueForStringParameter(this, "/srag/LANCE_TABLE");
    const modelEmbed   = ssm.StringParameter.valueForStringParameter(this, "/srag/MODEL_EMBED");
    const modelGen     = ssm.StringParameter.valueForStringParameter(this, "/srag/MODEL_GEN");
    const embedDim     = ssm.StringParameter.valueForStringParameter(this, "/srag/EMBED_DIM");
    const topK         = ssm.StringParameter.valueForStringParameter(this, "/srag/TOP_K");

    // 토글 & 옵션 (없으면 기본값으로 동작)
    const useWarming   = ssm.StringParameter.valueForStringParameter(this, "/srag/USE_WARMING", 1);
    const useDdbCache  = ssm.StringParameter.valueForStringParameter(this, "/srag/USE_DDB_CACHE", 1);
    const cacheTtlMin  = ssm.StringParameter.valueForStringParameter(this, "/srag/CACHE_TTL_MIN", 1);
    const useCtxCompr  = ssm.StringParameter.valueForStringParameter(this, "/srag/USE_CTX_COMPRESS", 1);
    const provider     = ssm.StringParameter.valueForStringParameter(this, "/srag/RETRIEVAL_PROVIDER", 1); // 'lance' | 'os'
    const osHost       = ssm.StringParameter.valueForStringParameter(this, "/srag/OS_HOST", 1);
    const osIndex      = ssm.StringParameter.valueForStringParameter(this, "/srag/OS_INDEX", 1);

    // ---------- S3 ----------
    const corpusBucket = s3.Bucket.fromBucketName(this, "CorpusBucket", bucketName);

    // ---------- VPC (A: 홉 최소화용 + S3 Gateway Endpoint) ----------
    const vpc = new ec2.Vpc(this, "RagVpc", {
      natGateways: 1,
      maxAzs: 2,
      subnetConfiguration: [
        { name: "public", subnetType: ec2.SubnetType.PUBLIC },
        { name: "private", subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      ],
    });
    vpc.addGatewayEndpoint("S3Gw", {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    // ---------- DDB 캐시 테이블 ----------
    const cacheTable = new ddb.Table(this, "RagQueryCache", {
      partitionKey: { name: "qhash", type: ddb.AttributeType.STRING },
      timeToLiveAttribute: "ttl",
      billingMode: ddb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ---------- Lambda (컨테이너) ----------
    const fn = new lambdaDocker.DockerImageFunction(this, "RagHandler", {
      code: lambdaDocker.DockerImageCode.fromImageAsset("../services/baseline_s"),
      memorySize: 3008,
      timeout: cdk.Duration.seconds(60),
      tracing: lambda.Tracing.ACTIVE,
      environment: {
        S3_BUCKET: bucketName,
        LANCE_URI: lanceUri,
        LANCE_TABLE: lanceTable,
        MODEL_EMBED: modelEmbed,
        MODEL_GEN: modelGen,
        EMBED_DIM: embedDim,
        TOP_K: topK,
        DDB_TABLE: cacheTable.tableName,
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    });

    corpusBucket.grantRead(fn);
    cacheTable.grantReadWriteData(fn);

    // Bedrock 권한
    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse"],
      resources: ["*"],
    }));
    // SSM 읽기 권한(토글 런타임 참조)
    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"],
      resources: ["*"],
    }));

    // ---------- API Gateway ----------
    const api = new apigw.RestApi(this, "RagApi", {
      deployOptions: {
        stageName: "prod",
        tracingEnabled: true,
        metricsEnabled: true,
        loggingLevel: apigw.MethodLoggingLevel.INFO,
        dataTraceEnabled: false,
      },
    });

    // 기존 query
    api.root.addResource("query").addMethod("POST", new apigw.LambdaIntegration(fn));

    // 기본 retrieve
    const retrieve = api.root.addResource("retrieve");
    retrieve.addMethod("GET",  new apigw.LambdaIntegration(fn));
    retrieve.addMethod("POST", new apigw.LambdaIntegration(fn));

    // Mock 전용 리소스 
    const retrieveMock = api.root.addResource("retrieve-mock");
    retrieveMock.addMethod("GET",  new apigw.LambdaIntegration(fn));
    retrieveMock.addMethod("POST", new apigw.LambdaIntegration(fn));

    // Baseline+ACE
    const retrieveAce = api.root.addResource("retrieve-mock-ace");
    retrieveAce.addMethod("GET", new apigw.LambdaIntegration(fn));
    retrieveAce.addMethod("POST", new apigw.LambdaIntegration(fn));

    // Baseline+OpenSearch (in-memory)
    const retrieveOsInMem = api.root.addResource("retrieve-os-inmem");
    retrieveOsInMem.addMethod("GET", new apigw.LambdaIntegration(fn));
    retrieveOsInMem.addMethod("POST", new apigw.LambdaIntegration(fn));

    // Baseline+OpenSearch (on-disk)
    const retrieveOsOnDisk = api.root.addResource("retrieve-os-ondisk");
    retrieveOsOnDisk.addMethod("GET", new apigw.LambdaIntegration(fn));
    retrieveOsOnDisk.addMethod("POST", new apigw.LambdaIntegration(fn));

    new cdk.CfnOutput(this, "ApiEndpoint", { value: api.url! });

    // ---------- 프리워밍 (USE_WARMING 토글이 true일 때만)
    if (useWarming === "true") {
      const rule = new events.Rule(this, "WarmRule", {
        schedule: events.Schedule.rate(cdk.Duration.minutes(5)),
      });
      rule.addTarget(new targets.LambdaFunction(fn, {
        event: cdk.aws_events.RuleTargetInput.fromObject({ warmup: true }),
      }));
    }
  }
}