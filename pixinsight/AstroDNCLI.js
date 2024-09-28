//https://pixinsight.com/forum/index.php?threads/scripting-documentation.10086/

#include <pjsr/DataType.jsh>
#include <../src/scripts/AdP/WCSmetadata.jsh>

#ifeq __PI_PLATFORM__ MACOSX
#define ASTRODNSCRPT_DIR File.homeDirectory + "/Library/Application Support/AstroDNScript"
#endif
#ifeq __PI_PLATFORM__ MSWINDOWS
#define ASTRODNSCRPT_DIR File.homeDirectory + "/AppData/Local/AstroDNScript"
#endif
#ifeq __PI_PLATFORM__ LINUX
#define ASTRODNSCRPT_DIR File.homeDirectory + "/.local/share/AstroDNScript"
#endif

#define SCRIPT_CONFIG ASTRODNSCRPT_DIR + "/AstroDNScript.json"

let astrodnParameters = {
   defaults: function() {
      return {
         model: "dist/v0.4.0-01",
         strength: 0.5,
         replaceTarget: true,
         device: "GPU",
         tiles: 3,
         stf: false,
         stfC: -2.8,
         stfB: 0.25
      }
   },

   isstf: function() {
      return astrodnParameters.stf;
   },

   saveToInstance: function() {
      Parameters.set("model", astrodnParameters.model);
      Parameters.set("strength", astrodnParameters.strength);
      Parameters.set("replaceTarget", astrodnParameters.replaceTarget);
      Parameters.set("device", astrodnParameters.device);
      Parameters.set("tiles", astrodnParameters.tiles);
      Parameters.set("stf", astrodnParameters.stf);
      Parameters.set("stfC", astrodnParameters.stfC);
      Parameters.set("stfB", astrodnParameters.stfB);      
   },

   loadFromInstance: function() {
      if (Parameters.has("model"))
         astrodnParameters.model = Parameters.getString("model");
      if (Parameters.has("strength"))
         astrodnParameters.strength = Parameters.getReal("strength");
      if (Parameters.has("replaceTarget"))
         astrodnParameters.replaceTarget = Parameters.getBoolean("replaceTarget");
      if (Parameters.has("device"))
         astrodnParameters.device = Parameters.getString("device");
      if (Parameters.has("tiles"))
         astrodnParameters.tiles = Parameters.getInteger("tiles");
      if (Parameters.has("stf"))
         astrodnParameters.stf = Parameters.getBoolean("stf");
      if (Parameters.has("stfC"))
         astrodnParameters.stfC = Parameters.getReal("stfC");
      if (Parameters.has("stfB"))
         astrodnParameters.stfB = Parameters.getReal("stfB");
   },

   nullishcoales: function(a,b) {
      return (a !== null && a !== undefined) ? a : b;
   },

   loadFromFile: function() {

      var params = undefined
      if (File.exists(SCRIPT_CONFIG)) {
         try {
            params = JSON.parse(File.readTextFile(SCRIPT_CONFIG));
         } catch (error) {
            Console.warningln("Loading AstroDN script settings failed...");
            Console.warningln(error);
         }
      }

      let defaults = astrodnParameters.defaults();
      // set default params
      if ( params == undefined ) {
         params = defaults;
      }

      astrodnParameters.replaceTarget = this.nullishcoales(params.replaceTarget,defaults.replaceTarget);
      astrodnParameters.strength = this.nullishcoales(params.strength,defaults.strength)
      astrodnParameters.model = this.nullishcoales(params.model,defaults.model);
      astrodnParameters.device = this.nullishcoales(params.device,defaults.device);
      astrodnParameters.tiles = this.nullishcoales(params.tiles,defaults.tiles);
      astrodnParameters.stf = this.nullishcoales(params.stf,defaults.stf);
      astrodnParameters.stfC = this.nullishcoales(params.stfC,defaults.stfC);
      astrodnParameters.stfB = this.nullishcoales(params.stfB,defaults.stfB);
   },

   saveToFile: function() {
      File.writeTextFile(SCRIPT_CONFIG, JSON.stringify(astrodnParameters));
   },

   init: function() {

      if (!File.directoryExists(ASTRODNSCRPT_DIR)) {
         File.createDirectory(ASTRODNSCRPT_DIR, true);
      }

      astrodnParameters.loadFromFile();

      astrodnParameters.loadFromInstance();
   },

   reset: function() {
		if ( File.exists(SCRIPT_CONFIG) ) {
			File.remove(SCRIPT_CONFIG);
		};

		// load preferences
		astrodnParameters.loadFromFile();
	}
};

function AstroDenoiseCLI() {

   function executeCLICommand(cmd) {

      Console.writeln("Executing CLI command " + cmd);

      let noError = true;

      this.process = new ExternalProcess;
      this.process.onStarted = function() {
         Console.noteln('Starting AstroDenoise...');
      };
      this.process.onError = function(code) {
         Console.criticalln('ERROR: ' + code);
      };
      this.process.onFinished = function() {
         Console.noteln('AstroDenoise completed.');
      }

      this.process.onStandardOutputDataAvailable = function() {
         Console.writeln(this.stdout.toString());
      };

      this.process.onStandardErrorDataAvailable  = function() {
         Console.criticalln('AstroDenoise Error: ' + this.stderr.toString());
      };

      try {

         this.process.start(cmd);
         for ( ; this.process.isStarting; )
            processEvents();
         for ( ; this.process.isRunning; )
            processEvents();

         return true;
      }
      catch(error) {
         Console.criticalln(error);
         return false;
      }
   }

   function getCLICommand(imagePath) {

      imagePath = File.unixPathToWindows(imagePath);
      //python -m D:\pydeep\astro-csbdeep\astrodenoise.main
      //var cmdLine = '"AstroDenoise" ' +
      var cmdLine = '"D:\\pydeep\\astro-csbdeep\\build\\AstroDenoise\\AstroDenoise" ' +
         '"' + imagePath + '"';

      if (astrodnParameters.strength != 0.5)
         cmdLine += ' --strength=' + astrodnParameters.strength;

      if (astrodnParameters.stf) {
         cmdLine += ' --normalize';
         cmdLine += ' --norm-C=' + astrodnParameters.stfC;
         cmdLine += ' --norm-B=' + astrodnParameters.stfB;
      }      

      cmdLine += ' --model=' + astrodnParameters.model;

      cmdLine += ' --device=' + astrodnParameters.device;

      cmdLine += ' --tiles=' + astrodnParameters.tiles;

      return cmdLine;
   }

   function assign(view, toView) {
      var P = new PixelMath;
      P.expression = view.id;
      P.useSingleExpression = true;
      P.clearImageCacheAndExit = false;
      P.cacheGeneratedImages = false;
      P.generateOutput = true;
      P.singleThreaded = false;
      P.optimization = true;
      P.use64BitWorkingImage = false;
      P.createNewImage = false;
      P.newImageColorSpace = PixelMath.prototype.SameAsTarget;
      P.newImageSampleFormat = PixelMath.prototype.SameAsTarget;

      P.executeOn(toView);
   }

   function cloneHidden(view, postfix, swapfile=true) {

      // Pick an unused name for the imageId
      var newId = null;
      if (ImageWindow.windowById(view.id + postfix).isNull)
         newId = view.id + postfix;
      else {
         for (var n = 1 ; n <= 99 ; n++) {
            if (ImageWindow.windowById(view.id + postfix + n).isNull) {
               newId = view.id + postfix + n;
               break;
            }
         }
      }
      if (newId == null) {
            (new MessageBox("Couldn't find a unique image name. Bailing out.",
                  TITLE, StdIcon_Error, StdButton_Ok)).execute();
            return;
      }

      var P = new PixelMath;
      P.expression = "$T";
      P.useSingleExpression = true;
      P.clearImageCacheAndExit = false;
      P.cacheGeneratedImages = false;
      P.generateOutput = true;
      P.singleThreaded = false;
      P.optimization = true;
      P.use64BitWorkingImage = false;
      P.rescale = false;
      P.truncate = true;
      P.createNewImage = true;
      P.showNewImage = false;
      P.newImageId = newId;
      P.newImageWidth = 0;
      P.newImageHeight = 0;
      P.newImageAlpha = false;
      P.newImageColorSpace = PixelMath.prototype.SameAsTarget;
      P.newImageSampleFormat = PixelMath.prototype.SameAsTarget;
      P.executeOn(view, swapfile);

      return View.viewById(P.newImageId);
   }

   function copyImagePathtoView(imagePath, view) {
      var windows = ImageWindow.open(imagePath);
      if (windows != null && windows.length > 0) {
         let firstWindow = windows[0];
         assign(firstWindow.mainView, view);
         firstWindow.forceClose();
      }
   }

   function getTempFile() {
      return File.systemTempDirectory + getFileSystemSeparator() + "astrodenoise_" + Math.round(Math.random()*10000)+ ".xisf";
   }

   function getFileSystemSeparator() {
      return corePlatform == "Windows" ? "\\" : "\/";
   }

   this.process = function () {

      console.show();

      var imagePath = getTempFile();
      Console.writeln('Temporary image file: ' + imagePath);
      var tempView = cloneHidden(this.targetView, "_SaveTemp");
      var saveResult = tempView.window.saveAs(imagePath, false, false, true, false)
      tempView.window.forceClose();

      if (!saveResult) {
         Console.warningln("Could not write file " + imagePath + " required to call AstroDN!");
         return;
      }

      if (executeCLICommand(getCLICommand(imagePath))) {

         var processedPath = imagePath.replace(".xisf", "_denoised.fits");
         processedPath = File.unixPathToWindows(processedPath);

         try {
            if (astrodnParameters.replaceTarget) {
               copyImagePathtoView(processedPath, this.targetView)
            }
            else {
               var newView = cloneHidden(this.targetView, "_AstroDN");
               let metadata = new ImageMetadata("AstroDN");
               metadata.ExtractMetadata(this.targetView.window);

               copyImagePathtoView(processedPath, newView)

               newView.window.keywords = this.targetView.window.keywords;
               if (!metadata.projection || !metadata.ref_I_G) {
                  Console.writeln("The image " + newView.id + " has no astrometric solution");
               }
               else {
                  metadata.SaveKeywords( newView.window, false);
                  metadata.SaveProperties( newView.window, TITLE + " " + VERSION);
               }
               newView.window.show();
            }
         }
         finally {
            if (File.exists(processedPath))
               File.remove(processedPath);
            File.remove(imagePath);
            console.hide();
         }
      }
      else {
         Console.criticalln("AstroDN failed!");
         File.remove(imagePath);
         console.show();
      }
   }
}

