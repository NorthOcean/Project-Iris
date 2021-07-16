<!--
 * @Author: Conghao Wong
 * @Date: 2021-04-16 09:07:48
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-07-16 16:53:48
 * @Description: file content
-->

# Classes Used in This Project

Packages:

```python
import modules.models.base as base
import modules.models.prediction as prediction

import tensorflow as tf
from tensorflow import keras
```
    

<!-- GRAPH BEGINS HERE -->
```mermaid
    graph LR
        builtins_object("object(builtins)") --> modules.models._base.writefunction_LogFunction("LogFunction(modules.models._base.writefunction)")
        builtins_object("object(builtins)") --> modules.models._base.visualization_Visualization("Visualization(modules.models._base.visualization)")
        modules.models._base.baseObject_BaseObject("BaseObject(modules.models._base.baseObject)") --> modules.models._base.structure_Structure("Structure(modules.models._base.structure)")
        builtins_object("object(builtins)") --> builtins_type("type(builtins)")
        argparse__AttributeHolder("_AttributeHolder(argparse)") --> argparse_Namespace("Namespace(argparse)")
        tensorflow.python.keras.engine.training_Model("Model(tensorflow.python.keras.engine.training)") --> modules.models._base.structure_Model("Model(modules.models._base.structure)")
        modules.models._base.baseObject_BaseObject("BaseObject(modules.models._base.baseObject)") --> modules.models._base.dataset._datasetManager_DatasetsManager("DatasetsManager(modules.models._base.dataset._datasetManager)")
        modules.models._base.baseObject_BaseObject("BaseObject(modules.models._base.baseObject)") --> modules.models._base.dataset._datasetManager_DatasetManager("DatasetManager(modules.models._base.dataset._datasetManager)")
        modules.models._base.args.base_BaseArgs("BaseArgs(modules.models._base.args.base)") --> modules.models._base.args.args_BaseArgsManager("BaseArgsManager(modules.models._base.args.args)")
        modules.models._base.baseObject_BaseObject("BaseObject(modules.models._base.baseObject)") --> modules.models._base.agent_Agent("Agent(modules.models._base.agent)")
        builtins_object("object(builtins)") --> modules.models._base.dataset._datasetInfo_DatasetInfo("DatasetInfo(modules.models._base.dataset._datasetInfo)")
        builtins_object("object(builtins)") --> modules.models._base.baseObject_BaseObject("BaseObject(modules.models._base.baseObject)")
        builtins_object("object(builtins)") --> modules.models._base.args.base_BaseArgs("BaseArgs(modules.models._base.args.base)")
        builtins_object("object(builtins)") --> modules.models._base.dataset._dataset_Dataset("Dataset(modules.models._base.dataset._dataset)")
        tqdm.utils_Comparable("Comparable(tqdm.utils)") --> tqdm.std_tqdm("tqdm(tqdm.std)")
        modules.models._base.visualization_Visualization("Visualization(modules.models._base.visualization)") --> modules.models._prediction.vis_TrajVisualization("TrajVisualization(modules.models._prediction.vis)")
        modules.models._prediction.agent_BaseAgentManager("BaseAgentManager(modules.models._prediction.agent)") --> modules.models._prediction.agent_TrainAgentManager("TrainAgentManager(modules.models._prediction.agent)")
        modules.models._base.dataset._datasetInfo_DatasetInfo("DatasetInfo(modules.models._base.dataset._datasetInfo)") --> modules.models._prediction.dataset._datasetManager_PredictionDatasetManager("PredictionDatasetManager(modules.models._prediction.dataset._datasetManager)")
        builtins_object("object(builtins)") --> modules.models._prediction.utils_Process("Process(modules.models._prediction.utils)")
        builtins_object("object(builtins)") --> modules.models._prediction.utils_Loss("Loss(modules.models._prediction.utils)")
        modules.models._base.baseObject_BaseObject("BaseObject(modules.models._base.baseObject)") --> modules.models._prediction.utils_IO("IO(modules.models._prediction.utils)")
        builtins_object("object(builtins)") --> modules.models._prediction.traj_EntireTrajectory("EntireTrajectory(modules.models._prediction.traj)")
        modules.models._base.structure_Structure("Structure(modules.models._base.structure)") --> modules.models._prediction.structure_Structure("Structure(modules.models._prediction.structure)")
        modules.models._base.dataset._datasetManager_DatasetsManager("DatasetsManager(modules.models._base.dataset._datasetManager)") --> modules.models._prediction.dataset._trainManager_DatasetsManager("DatasetsManager(modules.models._prediction.dataset._trainManager)")
        modules.models._base.dataset._datasetManager_DatasetManager("DatasetManager(modules.models._base.dataset._datasetManager)") --> modules.models._prediction.dataset._trainManager_DatasetManager("DatasetManager(modules.models._prediction.dataset._trainManager)")
        modules.models._base.args.args_BaseArgsManager("BaseArgsManager(modules.models._base.args.args)") --> modules.models._prediction.args_PredictionArgs("PredictionArgs(modules.models._prediction.args)")
        modules.models._base.structure_Model("Model(modules.models._base.structure)") --> modules.models._prediction.structure_Model("Model(modules.models._prediction.structure)")
        modules.models._base.baseObject_BaseObject("BaseObject(modules.models._base.baseObject)") --> modules.models._prediction.agent_MapManager("MapManager(modules.models._prediction.agent)")
        modules.models._base.agent_Agent("Agent(modules.models._base.agent)") --> modules.models._prediction.agent_BaseAgentManager("BaseAgentManager(modules.models._prediction.agent)")
        modules.models._base.visualization_Visualization("Visualization(modules.models._base.visualization)") --> modules.models._sceneModeling.vis_Visualization("Visualization(modules.models._sceneModeling.vis)")
        modules.models._base.agent_Agent("Agent(modules.models._base.agent)") --> modules.models._sceneModeling.agent_Agent("Agent(modules.models._sceneModeling.agent)")
        builtins_object("object(builtins)") --> modules.models._sceneModeling.utils_Data("Data(modules.models._sceneModeling.utils)")
        modules.models._prediction.dataset._trainManager_DatasetsManager("DatasetsManager(modules.models._prediction.dataset._trainManager)") --> modules.models._sceneModeling.trainManager_DatasetsManager("DatasetsManager(modules.models._sceneModeling.trainManager)")
        modules.models._base.dataset._datasetManager_DatasetManager("DatasetManager(modules.models._base.dataset._datasetManager)") --> modules.models._sceneModeling.trainManager_DatasetManager("DatasetManager(modules.models._sceneModeling.trainManager)")
        modules.models._base.structure_Structure("Structure(modules.models._base.structure)") --> modules.models._sceneModeling.structure_Structure("Structure(modules.models._sceneModeling.structure)")
        modules.models._base.structure_Model("Model(modules.models._base.structure)") --> modules.models._sceneModeling.structure_Model("Model(modules.models._sceneModeling.structure)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._utils_MultiHeadAttention("MultiHeadAttention(modules.applications._transformer._utils)")
        tensorflow.python.keras.engine.training_Model("Model(tensorflow.python.keras.engine.training)") --> modules.applications._transformer._transformer_Transformer("Transformer(modules.applications._transformer._transformer)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._transformer_EncoderLayer("EncoderLayer(modules.applications._transformer._transformer)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._transformer_Encoder("Encoder(modules.applications._transformer._transformer)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._transformer_DecoderLayer("DecoderLayer(modules.applications._transformer._transformer)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.applications._transformer._transformer_Decoder("Decoder(modules.applications._transformer._transformer)")
        builtins_object("object(builtins)") --> modules.Vertical._utils_Utils("Utils(modules.Vertical._utils)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.Vertical._layers_TrajEncoding("TrajEncoding(modules.Vertical._layers)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.Vertical._layers_LinearPrediction("LinearPrediction(modules.Vertical._layers)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.Vertical._layers_IFFTlayer("IFFTlayer(modules.Vertical._layers)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.Vertical._layers_GraphConv("GraphConv(modules.Vertical._layers)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.Vertical._layers_FFTlayer("FFTlayer(modules.Vertical._layers)")
        tensorflow.python.keras.engine.base_layer_Layer("Layer(tensorflow.python.keras.engine.base_layer)") --> modules.Vertical._layers_ContextEncoding("ContextEncoding(modules.Vertical._layers)")
        modules.models._prediction.args_PredictionArgs("PredictionArgs(modules.models._prediction.args)") --> modules.Vertical._args_VArgs("VArgs(modules.Vertical._args)")
        modules.models._prediction.structure_Model("Model(modules.models._prediction.structure)") --> modules.Vertical._VirisBeta_VIrisBetaModel("VIrisBetaModel(modules.Vertical._VirisBeta)")
        modules.models._prediction.structure_Structure("Structure(modules.models._prediction.structure)") --> modules.Vertical._VirisBeta_VIrisBeta("VIrisBeta(modules.Vertical._VirisBeta)")
        modules.models._prediction.structure_Model("Model(modules.models._prediction.structure)") --> modules.Vertical._VirisAlpha_VIrisAlphaModel("VIrisAlphaModel(modules.Vertical._VirisAlpha)")
        modules.models._prediction.structure_Structure("Structure(modules.models._prediction.structure)") --> modules.Vertical._VirisAlpha_VIrisAlpha("VIrisAlpha(modules.Vertical._VirisAlpha)")
        modules.Vertical._VirisAlpha_VIrisAlpha("VIrisAlpha(modules.Vertical._VirisAlpha)") --> modules.Vertical._Viris_VIris("VIris(modules.Vertical._Viris)")
        builtins_object("object(builtins)") --> modules.Vertical._Viris_BatchIndex("BatchIndex(modules.Vertical._Viris)")
        modules.models._prediction.structure_Model("Model(modules.models._prediction.structure)") --> modules.MSN._beta_G_MSNBeta_GModel("MSNBeta_GModel(modules.MSN._beta_G)")
        modules.models._prediction.structure_Structure("Structure(modules.models._prediction.structure)") --> modules.MSN._beta_G_MSNBeta_G("MSNBeta_G(modules.MSN._beta_G)")
        modules.models._prediction.args_PredictionArgs("PredictionArgs(modules.models._prediction.args)") --> modules.MSN._args_MSNArgs("MSNArgs(modules.MSN._args)")
        tensorflow.python.keras.engine.training_Model("Model(tensorflow.python.keras.engine.training)") --> modules.MSN._beta_G_Generator("Generator(modules.MSN._beta_G)")
        tensorflow.python.keras.engine.training_Model("Model(tensorflow.python.keras.engine.training)") --> modules.MSN._beta_G_Encoder("Encoder(modules.MSN._beta_G)")
        modules.models._prediction.structure_Model("Model(modules.models._prediction.structure)") --> modules.MSN._beta_D_MSNBeta_DModel("MSNBeta_DModel(modules.MSN._beta_D)")
        modules.models._prediction.structure_Structure("Structure(modules.models._prediction.structure)") --> modules.MSN._beta_D_MSNBeta_D("MSNBeta_D(modules.MSN._beta_D)")
        modules.models._prediction.structure_Model("Model(modules.models._prediction.structure)") --> modules.MSN._alpha_MSNAlphaModel("MSNAlphaModel(modules.MSN._alpha)")
        modules.models._prediction.structure_Structure("Structure(modules.models._prediction.structure)") --> modules.MSN._alpha_MSNAlpha("MSNAlpha(modules.MSN._alpha)")
        modules.MSN._alpha_MSNAlpha("MSNAlpha(modules.MSN._alpha)") --> modules.MSN._MSN_G_MSN_G("MSN_G(modules.MSN._MSN_G)")
        builtins_object("object(builtins)") --> modules.MSN._MSN_G_BatchIndex("BatchIndex(modules.MSN._MSN_G)")
        modules.MSN._alpha_MSNAlpha("MSNAlpha(modules.MSN._alpha)") --> modules.MSN._MSN_D_MSN_D("MSN_D(modules.MSN._MSN_D)")
```
<!-- GRAPH ENDS HERE -->
