<!--
 * @Author: Conghao Wong
 * @Date: 2021-04-16 09:07:48
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-05-25 09:20:22
 * @Description: file content
-->

# Classes Used in This Project

Packages:

```python
import modules.models.base as base
import modules.models.prediction as prediction

import modules.vgg as V

import tensorflow as tf
from tensorflow import keras
```
    

<!-- GRAPH BEGINS HERE -->
```mermaid
    graph LR
        modules.models._sceneModeling.training._trainingStructure_Structure("Structure(modules.models._sceneModeling.training._trainingStructure)") --> modules.vgg._vgg_vggmodel("vggmodel(modules.vgg._vgg)")
        builtins_object("object(builtins)") --> modules.models._base._writefunction_LogFunction("LogFunction(modules.models._base._writefunction)")
        modules.models._prediction.dataset._trainManager_DatasetsManager("DatasetsManager(modules.models._prediction.dataset._trainManager)") --> modules.models._sceneModeling.dataset._trainManager_DatasetsManager("DatasetsManager(modules.models._sceneModeling.dataset._trainManager)")
        modules.models._base.dataset._datasetManager_DatasetManager("DatasetManager(modules.models._base.dataset._datasetManager)") --> modules.models._sceneModeling.dataset._trainManager_DatasetManager("DatasetManager(modules.models._sceneModeling.dataset._trainManager)")
        builtins_object("object(builtins)") --> modules.models._base.args._argManager_BaseArgsManager("BaseArgsManager(modules.models._base.args._argManager)")
        modules.models._base.agent._agent_Agent("Agent(modules.models._base.agent._agent)") --> modules.models._sceneModeling.agent._agent_Agent("Agent(modules.models._sceneModeling.agent._agent)")
        builtins_object("object(builtins)") --> builtins_type("type(builtins)")
        modules.models._base.dataset._datasetInfo_DatasetInfo("DatasetInfo(modules.models._base.dataset._datasetInfo)") --> modules.models._prediction.dataset._datasetManager_PredictionDatasetManager("PredictionDatasetManager(modules.models._prediction.dataset._datasetManager)")
        modules.models._prediction.args._argManager_BasePredictArgs("BasePredictArgs(modules.models._prediction.args._argManager)") --> modules.models._prediction.args._argManager_TrainArgsManager("TrainArgsManager(modules.models._prediction.args._argManager)")
        modules.models._sceneModeling.training._trainingStructure_Model("Model(modules.models._sceneModeling.training._trainingStructure)") --> modules.vgg._vgg_score_VggSceneRegModel("VggSceneRegModel(modules.vgg._vgg_score)")
        modules.models._sceneModeling.training._trainingStructure_Structure("Structure(modules.models._sceneModeling.training._trainingStructure)") --> modules.vgg._vgg_score_VggRegModel("VggRegModel(modules.vgg._vgg_score)")
        builtins_object("object(builtins)") --> modules.models._base.vis._visualization_Visualization("Visualization(modules.models._base.vis._visualization)")
        tqdm.utils_Comparable("Comparable(tqdm.utils)") --> tqdm.std_tqdm("tqdm(tqdm.std)")
        modules.models._base._baseObject_BaseObject("BaseObject(modules.models._base._baseObject)") --> modules.models._base.training._trainingStructure_Structure("Structure(modules.models._base.training._trainingStructure)")
        modules.models._base._baseObject_BaseObject("BaseObject(modules.models._base._baseObject)") --> modules.models._base.dataset._datasetManager_DatasetsManager("DatasetsManager(modules.models._base.dataset._datasetManager)")
        modules.models._base._baseObject_BaseObject("BaseObject(modules.models._base._baseObject)") --> modules.models._base.dataset._datasetManager_DatasetManager("DatasetManager(modules.models._base.dataset._datasetManager)")
        modules.models._base._baseObject_BaseObject("BaseObject(modules.models._base._baseObject)") --> modules.models._base.agent._agent_Agent("Agent(modules.models._base.agent._agent)")
        builtins_object("object(builtins)") --> modules.models._base.dataset._datasetInfo_DatasetInfo("DatasetInfo(modules.models._base.dataset._datasetInfo)")
        tensorflow.python.keras.engine.training_Model("Model(tensorflow.python.keras.engine.training)") --> modules.models._base.training._trainingStructure_Model("Model(modules.models._base.training._trainingStructure)")
        builtins_object("object(builtins)") --> modules.models._base._baseObject_BaseObject("BaseObject(modules.models._base._baseObject)")
        builtins_object("object(builtins)") --> modules.models._base.dataset._dataset_Dataset("Dataset(modules.models._base.dataset._dataset)")
        modules.models._base.vis._visualization_Visualization("Visualization(modules.models._base.vis._visualization)") --> modules.models._prediction.vis._trajVisual_TrajVisualization("TrajVisualization(modules.models._prediction.vis._trajVisual)")
        modules.models._prediction.agent._agentManager_BaseAgentManager("BaseAgentManager(modules.models._prediction.agent._agentManager)") --> modules.models._prediction.agent._agentManager_TrainAgentManager("TrainAgentManager(modules.models._prediction.agent._agentManager)")
        typing__Final("_Final(typing)") --> typing_TypeVar("TypeVar(typing)")
        modules.models._base.training._trainingStructure_Structure("Structure(modules.models._base.training._trainingStructure)") --> modules.models._prediction.training._trainingStructure_Structure("Structure(modules.models._prediction.training._trainingStructure)")
        modules.models._base.dataset._datasetManager_DatasetsManager("DatasetsManager(modules.models._base.dataset._datasetManager)") --> modules.models._prediction.dataset._trainManager_DatasetsManager("DatasetsManager(modules.models._prediction.dataset._trainManager)")
        modules.models._base.dataset._datasetManager_DatasetManager("DatasetManager(modules.models._base.dataset._datasetManager)") --> modules.models._prediction.dataset._trainManager_DatasetManager("DatasetManager(modules.models._prediction.dataset._trainManager)")
        builtins_object("object(builtins)") --> modules.models._prediction._utils_Process("Process(modules.models._prediction._utils)")
        modules.models._base.training._trainingStructure_Model("Model(modules.models._base.training._trainingStructure)") --> modules.models._prediction.training._trainingStructure_Model("Model(modules.models._prediction.training._trainingStructure)")
        builtins_object("object(builtins)") --> modules.models._prediction._utils_Loss("Loss(modules.models._prediction._utils)")
        modules.models._base._baseObject_BaseObject("BaseObject(modules.models._base._baseObject)") --> modules.models._prediction._utils_IO("IO(modules.models._prediction._utils)")
        builtins_object("object(builtins)") --> modules.models._prediction.training._entireTraj_EntireTrajectory("EntireTrajectory(modules.models._prediction.training._entireTraj)")
        modules.models._base._baseObject_BaseObject("BaseObject(modules.models._base._baseObject)") --> modules.models._prediction.agent._agentManager_MapManager("MapManager(modules.models._prediction.agent._agentManager)")
        modules.models._prediction.args._argManager_TrainArgsManager("TrainArgsManager(modules.models._prediction.args._argManager)") --> modules.models._prediction.args._argManager_OnlineArgsManager("OnlineArgsManager(modules.models._prediction.args._argManager)")
        modules.models._base.args._argManager_BaseArgsManager("BaseArgsManager(modules.models._base.args._argManager)") --> modules.models._prediction.args._argManager_BasePredictArgs("BasePredictArgs(modules.models._prediction.args._argManager)")
        modules.models._prediction.agent._agentManager_BaseAgentManager("BaseAgentManager(modules.models._prediction.agent._agentManager)") --> modules.models._prediction.agent._agentManager_OnlineAgentManager("OnlineAgentManager(modules.models._prediction.agent._agentManager)")
        modules.models._base.agent._agent_Agent("Agent(modules.models._base.agent._agent)") --> modules.models._prediction.agent._agentManager_BaseAgentManager("BaseAgentManager(modules.models._prediction.agent._agentManager)")
        modules.models._base.vis._visualization_Visualization("Visualization(modules.models._base.vis._visualization)") --> modules.models._sceneModeling.vis._sceneVisual_Visualization("Visualization(modules.models._sceneModeling.vis._sceneVisual)")
        modules.models._base.training._trainingStructure_Structure("Structure(modules.models._base.training._trainingStructure)") --> modules.models._sceneModeling.training._trainingStructure_Structure("Structure(modules.models._sceneModeling.training._trainingStructure)")
        modules.models._base.training._trainingStructure_Model("Model(modules.models._base.training._trainingStructure)") --> modules.models._sceneModeling.training._trainingStructure_Model("Model(modules.models._sceneModeling.training._trainingStructure)")
        builtins_object("object(builtins)") --> modules.models._sceneModeling._utils_Data("Data(modules.models._sceneModeling._utils)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.models._helpmethods._helpmethods_TrainableAdjMatrix("TrainableAdjMatrix(modules.models._helpmethods._helpmethods)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.models._helpmethods._helpmethods_GraphConv("GraphConv(modules.models._helpmethods._helpmethods)")
        modules.models._sceneModeling.training._trainingStructure_Model("Model(modules.models._sceneModeling.training._trainingStructure)") --> modules.vgg._vgg_VggSceneModel("VggSceneModel(modules.vgg._vgg)")
        modules.models._sceneModeling.training._trainingStructure_Structure("Structure(modules.models._sceneModeling.training._trainingStructure)") --> modules.vgg._draw_gt_DrawSceneGT("DrawSceneGT(modules.vgg._draw_gt)")
        modules.models._prediction.training._trainingStructure_Structure("Structure(modules.models._prediction.training._trainingStructure)") --> modules.transformerPrediction._t_Structure("Structure(modules.transformerPrediction._t)")
        modules.models._prediction.training._trainingStructure_Model("Model(modules.models._prediction.training._trainingStructure)") --> modules.transformerPrediction._t_Model("Model(modules.transformerPrediction._t)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._utils_MultiHeadAttention("MultiHeadAttention(modules.applications._transformer._utils)")
        tensorflow.python.keras.engine.training_Model("Model(tensorflow.python.keras.engine.training)") --> modules.applications._transformer._transformer_Transformer("Transformer(modules.applications._transformer._transformer)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._transformer_EncoderLayer("EncoderLayer(modules.applications._transformer._transformer)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._transformer_Encoder("Encoder(modules.applications._transformer._transformer)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._transformer_DecoderLayer("DecoderLayer(modules.applications._transformer._transformer)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._transformer_Decoder("Decoder(modules.applications._transformer._transformer)")
        modules.vgg._vgg_score_VggRegModel("VggRegModel(modules.vgg._vgg_score)") --> modules.satoshiVGG._model_SVStructure("SVStructure(modules.satoshiVGG._model)")
        modules.models._sceneModeling.dataset._trainManager_DatasetsManager("DatasetsManager(modules.models._sceneModeling.dataset._trainManager)") --> modules.satoshiVGG._model_DatasetsManager("DatasetsManager(modules.satoshiVGG._model)")
        modules.models._sceneModeling.dataset._trainManager_DatasetManager("DatasetManager(modules.models._sceneModeling.dataset._trainManager)") --> modules.satoshiVGG._model_DatasetManager("DatasetManager(modules.satoshiVGG._model)")
        modules.satoshiVGG._model_SVStructure("SVStructure(modules.satoshiVGG._model)") --> modules.satoshiVGG._newNet_ResNetRegStructure("ResNetRegStructure(modules.satoshiVGG._newNet)")
        modules.models._sceneModeling.training._trainingStructure_Model("Model(modules.models._sceneModeling.training._trainingStructure)") --> modules.satoshiVGG._newNet_ResNetRegModel("ResNetRegModel(modules.satoshiVGG._newNet)")
        modules.vgg._vgg_score_VggRegModel("VggRegModel(modules.vgg._vgg_score)") --> modules.satoshiVGG._alpha_agent_AlphaStructure("AlphaStructure(modules.satoshiVGG._alpha_agent)")
        modules.models._sceneModeling.dataset._trainManager_DatasetsManager("DatasetsManager(modules.models._sceneModeling.dataset._trainManager)") --> modules.satoshiVGG._alpha_agent_AlphaDatasetsManager("AlphaDatasetsManager(modules.satoshiVGG._alpha_agent)")
        modules.models._sceneModeling.dataset._trainManager_DatasetManager("DatasetManager(modules.models._sceneModeling.dataset._trainManager)") --> modules.satoshiVGG._alpha_agent_AlphaDatasetManager("AlphaDatasetManager(modules.satoshiVGG._alpha_agent)")
        modules.models._sceneModeling.agent._agent_Agent("Agent(modules.models._sceneModeling.agent._agent)") --> modules.satoshiVGG._alpha_agent_AlphaAgent("AlphaAgent(modules.satoshiVGG._alpha_agent)")
        modules.models._prediction.training._trainingStructure_Structure("Structure(modules.models._prediction.training._trainingStructure)") --> modules.satoshi._beta_transformer_SatoshiBetaTransformer("SatoshiBetaTransformer(modules.satoshi._beta_transformer)")
        modules.models._prediction.training._trainingStructure_Structure("Structure(modules.models._prediction.training._trainingStructure)") --> modules.satoshi._beta_SatoshiBeta("SatoshiBeta(modules.satoshi._beta)")
        modules.models._prediction.training._trainingStructure_Structure("Structure(modules.models._prediction.training._trainingStructure)") --> modules.satoshi._alpha_transformer_SatoshiAlphaTransformer("SatoshiAlphaTransformer(modules.satoshi._alpha_transformer)")
        modules.models._prediction.training._trainingStructure_Structure("Structure(modules.models._prediction.training._trainingStructure)") --> modules.satoshi._alpha_SatoshiAlpha("SatoshiAlpha(modules.satoshi._alpha)")
        modules.satoshi._alpha_transformer_SatoshiAlphaTransformer("SatoshiAlphaTransformer(modules.satoshi._alpha_transformer)") --> modules.satoshi._satoshi_transformer_SatoshiTransformer("SatoshiTransformer(modules.satoshi._satoshi_transformer)")
        modules.models._prediction.args._argManager_TrainArgsManager("TrainArgsManager(modules.models._prediction.args._argManager)") --> modules.satoshi._args_SatoshiArgs("SatoshiArgs(modules.satoshi._args)")
        modules.models._prediction.training._trainingStructure_Model("Model(modules.models._prediction.training._trainingStructure)") --> modules.satoshi._alpha_transformer_SatoshiAlphaTransformerModel("SatoshiAlphaTransformerModel(modules.satoshi._alpha_transformer)")
        modules.models._base.training._trainingStructure_Model("Model(modules.models._base.training._trainingStructure)") --> modules.satoshi._beta_SatoshiBetaModel("SatoshiBetaModel(modules.satoshi._beta)")
        modules.models._base.training._trainingStructure_Model("Model(modules.models._base.training._trainingStructure)") --> modules.satoshi._alpha_SatoshiAlphaModel("SatoshiAlphaModel(modules.satoshi._alpha)")
        modules.satoshi._alpha_SatoshiAlpha("SatoshiAlpha(modules.satoshi._alpha)") --> modules.satoshi._satoshi_Satoshi("Satoshi(modules.satoshi._satoshi)")
        modules.models._prediction.training._trainingStructure_Model("Model(modules.models._prediction.training._trainingStructure)") --> modules.satoshi._beta_transformer_SatoshiBetaTransformerModel("SatoshiBetaTransformerModel(modules.satoshi._beta_transformer)")
        modules.satoshi._args_SatoshiArgs("SatoshiArgs(modules.satoshi._args)") --> modules.satoshi._args_SatoshiOnlineArgs("SatoshiOnlineArgs(modules.satoshi._args)")
        modules.models._base.training._trainingStructure_Model("Model(modules.models._base.training._trainingStructure)") --> modules.linear._linear_LinearModel("LinearModel(modules.linear._linear)")
        modules.models._prediction.training._trainingStructure_Structure("Structure(modules.models._prediction.training._trainingStructure)") --> modules.linear._linear_Linear("Linear(modules.linear._linear)")
```
<!-- GRAPH ENDS HERE -->
