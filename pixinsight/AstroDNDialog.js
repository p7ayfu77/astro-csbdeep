#include <pjsr/StdIcon.jsh>
#include <pjsr/StdButton.jsh>
#include <pjsr/StdDialogCode.jsh>
#include <pjsr/Sizer.jsh>
#include <pjsr/FrameStyle.jsh>
#include <pjsr/TextAlign.jsh>
#include <pjsr/ColorSpace.jsh>
#include <pjsr/NumericControl.jsh>
#include <pjsr/CheckState.jsh>
#include <pjsr/SampleType.jsh>

#define VERSION "0.5.7"
#define TITLE  "AstroDenoise"
#define TEXT   "Remove noise from astrophotography images using AstroDenoise CLI. Please ensure script version matches AstroDenoise version!"

#define FORMWIDTH 360

function AstroDNDialog(cli) {
   this.__base__ = Dialog
   this.__base__();

   this.userResizable = false;
   this.scaledMinWidth = FORMWIDTH;

   this.helpLabel = new Label(this);
   this.helpLabel.frameStyle = FrameStyle_Box;
   this.helpLabel.margin = 4;
   this.helpLabel.wordWrapping = true;
   this.helpLabel.useRichText = true;
   this.helpLabel.text = "<b>" + TITLE + " Script v"
      + VERSION + "</b> &mdash; "
      + TEXT;

   this.imageViewSelectorFrame = new Frame(this);
   this.imageViewSelectLabel = new Label(this);
   this.imageViewSelectLabel.text = "Image:";
   this.imageViewSelectLabel.toolTip = "<p>Select the image for denoising.</p>";
   this.imageViewSelectLabel.textAlignment = TextAlign_Left | TextAlign_VertCenter;

   this.imageViewSelector = new ViewList(this);
   this.imageViewSelector.maxWidth = FORMWIDTH;
   this.imageViewSelector.getMainViews();

   with (this.imageViewSelectorFrame) {
      sizer = new HorizontalSizer();

      with (sizer) {
         margin = 6;
         add(this.imageViewSelectLabel);
         addSpacing(8);
         add(this.imageViewSelector);
         adjustToContents();
      }
   }

   this.imageViewSelector.onViewSelected = function (view) {
      cli.targetView = view;
   };

   this.modelSelectionFrame = new Frame(this);
   this.modelSelectionLabel = new Label(this);
   this.modelSelectionLabel.text = "Model:";
   this.modelSelectionLabel.tooltip = "<p>The selected AI denoise model.</p>";
   this.modelSelectionLabel.textAlignment = TextAlign_Left | TextAlign_VertCenter;

   this.modelSelectionList = new ComboBox(this);
   this.modelSelectionList.addItem("dist/v0.3.0-01");
   this.modelSelectionList.addItem("dist/v0.4.0-01");
   this.modelSelectionList.addItem("dist/v0.4.0-02");
   this.modelSelectionList.addItem("dist/v0.5.0-01");

   with (this.modelSelectionFrame) {
      sizer = new HorizontalSizer();

      with (sizer) {
         margin = 6;

         add(this.modelSelectionLabel);
         addSpacing(8);
         add(this.modelSelectionList);
         adjustToContents();
      }
   }
   
   this.modelSelectionList.currentItem = this.modelSelectionList.findItem(astrodnParameters.model);

   this.modelSelectionList.onItemSelected = function (index) {
      astrodnParameters.model = this.itemText(index);
   }

   // Strength
   this.strengthSlider = new NumericControl(this);
   this.strengthSlider.label.text = "Strength:";
   this.strengthSlider.toolTip = "<p>Increase or decrease the denoise strength.</p>";
   this.strengthSlider.setRange(0.0, 1.0);
   this.strengthSlider.slider.setRange(0.0, 1000.0);
   this.strengthSlider.setPrecision(3);
   this.strengthSlider.setReal(true);
   this.strengthSlider.setValue(astrodnParameters.strength);

   this.strengthSlider.onValueUpdated = function (t) {
      astrodnParameters.strength = t;
   }

   this.resetStrengthButton = new ToolButton(this);
   this.resetStrengthButton.icon = this.scaledResource(":/icons/clear-inverted.png");
   this.resetStrengthButton.setScaledFixedSize(24, 24);
   this.resetStrengthButton.toolTip = "<p>Reset denoising strength.</p>";
   this.resetStrengthButton.onClick = () => {
      astrodnParameters.strength = 0.5;
      this.strengthSlider.setValue(0.5);
   }

   this.strengthControl = new HorizontalSizer();
   this.strengthControl.maxWidth = FORMWIDTH;
   this.strengthControl.margin = 6;
   this.strengthControl.add(this.strengthSlider);
   this.strengthControl.add(this.resetStrengthButton);

   ////////////////////////

   this.stfCSlider = new NumericControl(this);
   this.stfCSlider.label.text = "STF Low Clipping:";
   this.stfCSlider.toolTip = "<p>STF Stretch C (Low Clipping)</p>";
   this.stfCSlider.setRange(-4.0, 0.0);
   this.stfCSlider.slider.setRange(0.0, 1000.0);
   this.stfCSlider.setPrecision(3);
   this.stfCSlider.setReal(true);
   this.stfCSlider.setValue(astrodnParameters.stfC);

   this.stfCSlider.onValueUpdated = function (t) {
      astrodnParameters.stfC = t;
   }

   this.stfCResetButton = new ToolButton(this);
   this.stfCResetButton.icon = this.scaledResource(":/icons/clear-inverted.png");
   this.stfCResetButton.setScaledFixedSize(24, 24);
   this.stfCResetButton.toolTip = "<p>Reset denoising strength.</p>";
   this.stfCResetButton.onClick = () => {
      astrodnParameters.stfC = -2.8;
      this.stfCSlider.setValue(-2.8);
   }

   this.stfCControl = new Control( this );
   this.stfCControl.sizer = new HorizontalSizer();
   this.stfCControl.sizer.maxWidth = FORMWIDTH;
   this.stfCControl.sizer.margin = 6;
   this.stfCControl.sizer.add(this.stfCSlider);
   this.stfCControl.sizer.add(this.stfCResetButton);
   this.stfCControl.enabled = astrodnParameters.isstf();

   ///////////////////////

   this.stfBSlider = new NumericControl(this);
   this.stfBSlider.label.text = "STF Strength:";
   this.stfBSlider.toolTip = "<p>STF Stretch B (Strength)</p>";
   this.stfBSlider.setRange(0.0, 1.0);
   this.stfBSlider.slider.setRange(0.0, 1000.0);
   this.stfBSlider.setPrecision(3);
   this.stfBSlider.setReal(true);
   this.stfBSlider.setValue(astrodnParameters.stfB);

   this.stfBSlider.onValueUpdated = function (t) {
      astrodnParameters.stfB = t;
   }

   this.stfBResetButton = new ToolButton(this);
   this.stfBResetButton.icon = this.scaledResource(":/icons/clear-inverted.png");
   this.stfBResetButton.setScaledFixedSize(24, 24);
   this.stfBResetButton.toolTip = "<p>Reset denoising strength.</p>";
   this.stfBResetButton.onClick = () => {
      astrodnParameters.stfB = 0.25;
      this.stfBSlider.setValue(0.25);
   }

   this.stfBControl = new Control( this );
   this.stfBControl.sizer = new HorizontalSizer();
   this.stfBControl.sizer.maxWidth = FORMWIDTH;
   this.stfBControl.sizer.margin = 6;
   this.stfBControl.sizer.add(this.stfBSlider);
   this.stfBControl.sizer.add(this.stfBResetButton);
   this.stfBControl.enabled = astrodnParameters.isstf();

   ////////////////////////
   // Process with STF
   this.processSTF = new Frame;
   this.processSTF.sizer = new HorizontalSizer;
   this.processSTF.sizer.margin = 6;
   this.processSTF.sizer.spacing = 6;

   this.processSTF.sizer.addStretch();

   this.processSTFCheckbox = new CheckBox(this);
   this.processSTFCheckbox.text = "Pre-process with STF";
   this.processSTFCheckbox.checked = astrodnParameters.stf;
   this.processSTFCheckbox.toolTip = "<p>For linear images, pre-process the image to denoise with STF stretch. The denoise process is best executed on non-linear images.</p>";
   this.processSTF.sizer.add(this.processSTFCheckbox);

   this.processSTFCheckbox.onCheck = function (checked) {
      astrodnParameters.stf = checked;
      this.dialog.stfCControl.enabled = astrodnParameters.isstf();
      this.dialog.stfBControl.enabled = astrodnParameters.isstf();
   }

   // Replace target view
   this.replaceTargetFrame = new Frame;
   this.replaceTargetFrame.sizer = new HorizontalSizer;
   this.replaceTargetFrame.sizer.margin = 6;
   this.replaceTargetFrame.sizer.spacing = 6;

   this.replaceTargetFrame.sizer.addStretch();

   this.replaceTargetCheckbox = new CheckBox(this);
   this.replaceTargetCheckbox.text = "Replace the target view";
   this.replaceTargetCheckbox.checked = astrodnParameters.replaceTarget;
   this.replaceTargetCheckbox.toolTip = "<p>Replaces the target view with the processed image, if checked. Otherwise, a new image will be created.</p>";
   this.replaceTargetFrame.sizer.add(this.replaceTargetCheckbox);

   this.replaceTargetCheckbox.onCheck = function (checked) {
      astrodnParameters.replaceTarget = checked;
   }

   this.buttonFrame = new Frame;

   this.buttonFrame.sizer = new HorizontalSizer;
   this.buttonFrame.sizer.margin = 6;
   this.buttonFrame.sizer.spacing = 6;

   this.newInstanceButton = new ToolButton(this);
   this.newInstanceButton.icon = this.scaledResource(":/process-interface/new-instance.png");
   this.newInstanceButton.setScaledFixedSize(24, 24);
   this.newInstanceButton.onMousePress = () => {
      astrodnParameters.saveToInstance();
      Console.hide();
      this.newInstance();
   }

   this.ok_Button = new ToolButton(this);
   this.ok_Button.icon = this.scaledResource(":/process-interface/execute.png");
   this.ok_Button.setScaledFixedSize(24, 24);
   this.ok_Button.toolTip = "<p>Execute.</p>";
   this.ok_Button.onClick = () => {
      astrodnParameters.saveToFile();
      this.ok();
      cli.process();
   };

   this.cancel_Button = new ToolButton(this);
   this.cancel_Button.icon = this.scaledResource(":/process-interface/cancel.png");
   this.cancel_Button.setScaledFixedSize(24, 24);
   this.cancel_Button.toolTip = "<p>Close this dialog with no changes.</p>";
   this.cancel_Button.onClick = () => {
      this.cancel();
   };

   this.help_Button = new ToolButton(this);
   this.help_Button.icon = this.scaledResource(":/process-interface/browse-documentation.png");
   this.help_Button.setScaledFixedSize(24, 24);
   this.help_Button.toolTip = "<p>Shows the script documentation.</p>";
   this.help_Button.onClick = () => {
      Dialog.browseScriptDocumentation("AstroDN");
   };

   this.reset_Button = new ToolButton(this);
   this.reset_Button.icon = this.scaledResource(":/process-interface/reset.png");
   this.reset_Button.setScaledFixedSize(24, 24);
   this.reset_Button.toolTip = "<p>Resets all settings to their defaults.</p>";
   this.reset_Button.onClick = () => {
      astrodnParameters.reset();
      this.dialog.modelSelectionList.currentItem = this.dialog.modelSelectionList.findItem(astrodnParameters.model);
      this.dialog.strengthSlider.setValue(astrodnParameters.strength);
      this.processSTFCheckbox.checked = astrodnParameters.stf;
      this.dialog.stfCSlider.setValue(astrodnParameters.stfC);
      this.dialog.stfBSlider.setValue(astrodnParameters.stfB);
      this.dialog.replaceTargetCheckbox.checked = astrodnParameters.replaceTarget;
   }

   this.buttonFrame.sizer.add(this.newInstanceButton);
   this.buttonFrame.sizer.addSpacing(8);
   this.buttonFrame.sizer.add(this.ok_Button);
   this.buttonFrame.sizer.addSpacing(8);
   this.buttonFrame.sizer.add(this.cancel_Button);
   this.buttonFrame.sizer.addSpacing(32);
   this.buttonFrame.sizer.add(this.help_Button);
   this.buttonFrame.sizer.addSpacing(16);
   this.buttonFrame.sizer.add(this.reset_Button);

   this.sizer = new VerticalSizer;
   this.sizer.margin = 8;

   this.sizer.add(this.helpLabel);
   this.sizer.addSpacing(8);

   this.sizer.add(this.imageViewSelectorFrame);
   this.sizer.addSpacing(4);
   this.sizer.add(this.modelSelectionFrame);
   this.sizer.add(this.strengthControl);
   this.sizer.addSpacing(4);
   this.sizer.add(this.processSTF);
   this.sizer.add(this.stfCControl);
   this.sizer.add(this.stfBControl);
   this.sizer.addSpacing(4);
   this.sizer.add(this.replaceTargetFrame);
   this.sizer.addSpacing(16);

   this.sizer.add(this.buttonFrame);

   if (cli.targetView !== undefined) {
      this.imageViewSelector.currentView = cli.targetView;
   }
}

AstroDNDialog.prototype = new Dialog