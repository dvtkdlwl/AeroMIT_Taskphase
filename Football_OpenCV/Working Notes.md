# Working Notes

*The following are working notes I wrote while fine-tuning my thought process for the project.*

first eliminating the pitch as well as all the white lines (this isolates the players ENTIRELY).
this resultant image is then fed into a generalcontours function which effectively identifies all the separate players from the audience. it plots these coordinates onto a pure black frame as white rectangular blobs.
then, we colour-mask red players and isolate them in order to use bitwise_and on this image as well as the pure black frame's blobs.
we do the same for black players- colour-mask + isolate + plot them on the black frame and draw squares on the bitwise_and result.

finally, for the zoomed-in shots, we need a new approach. 
this is because the players are significantly bigger in these frames. my normal area thresholds are failing to capture them. in fact, the red mask seems to work perfectly fine in this case, but the without_pitch frame is performing terribly because it is capturing literally everything.

The red isolation works when I do it while merging the without_pitch frame and the red filtered frame. Clearly, there is some issue with the area thresholding in the pureblackframe because it is not plotting blobs correctly on the input frame.

The change must be made in the generalgetcontours function call- i am calling it on without_pitch which is sheer abuse of the area filter. it must be changed.

Best solution- eliminate white boundary lines better!- done

hm now for red team zoomed-in shots, the issue is different. players are often surrounded by white background... without_pitch fails because white dominates the scene. red masking still works reliably.

let's try to rely more heavily on color-based isolation?
treat contour detection as secondary validation rather than the primary filter!- done.
