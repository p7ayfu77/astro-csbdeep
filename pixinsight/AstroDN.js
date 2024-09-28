#feature-id AstroDenoise : Toolbox > AstroDenoise
#feature-info  A script to run AstroDenoise from within PixInsight.

#include "AstroDNCLI.js"
#include "AstroDNDialog.js"

function main() {
   console.hide();

   if (Parameters.isViewTarget) {
      var targetView = Parameters.targetView
   }
   else {
      var targetView = ImageWindow.activeWindow.currentView;
   }

   if ( !targetView || !targetView.id ) {
		let mb = new MessageBox(
				"<p><center>No valid view is selected.</center></p>",
				TITLE,
				StdIcon_NoIcon,
				StdButton_Ok
		);
		mb.execute()
		return
	}

   astrodnParameters.init();
   	
   let astroDenoiseCLI = new AstroDenoiseCLI();
   astroDenoiseCLI.targetView = targetView;

	if (Parameters.isViewTarget) {
      astroDenoiseCLI.process();
   }
   else {
      let dialog = new AstroDNDialog(astroDenoiseCLI);
      dialog.execute();
   };
}

main();