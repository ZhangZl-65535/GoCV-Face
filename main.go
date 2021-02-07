package main

import (
	"fmt"
	"image"
	"image/color"
	"gocv.io/x/gocv"
)

func main() {
	cam, err := gocv.VideoCaptureDevice(0)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cam.Close()

	window := gocv.NewWindow("Face Detect")
	defer window.Close()
	window.SetWindowProperty(gocv.WindowPropertyAspectRatio, gocv.WindowNormal)

	// load classifier to recognize faces
	faceclassifier := gocv.NewCascadeClassifier()
	defer faceclassifier.Close()

	if !faceclassifier.Load("./haarcascade_frontalface_alt2.xml") {
		fmt.Println("read face cascade file error")
		return
	}

	// load classifier to recognize eyes
	eyeclassifier := gocv.NewCascadeClassifier()
	defer eyeclassifier.Close()

	if !eyeclassifier.Load("./haarcascade_eye_tree_eyeglasses.xml") {
		fmt.Println("read eye cascade file error")
		return
	}

	img := gocv.NewMat()
	defer img.Close()

	clr := color.RGBA{0xCC, 0x66, 0x0, 0x0}

	for {
		if ok := cam.Read(&img); !ok {
			fmt.Println("cannot read device")
			break
		}
		if img.Empty() {
			continue
		}

		// detect faces
		faceRects := faceclassifier.DetectMultiScale(img)
		if len(faceRects) > 0 {
			fmt.Printf("found %d faces\n", len(faceRects))
			rects := []image.Rectangle{}
			for idx, r := range faceRects {
				faceMat := img.Region(r)
				// detect eyes
				eyeRects := eyeclassifier.DetectMultiScale(faceMat)
				fmt.Printf("face %d: eyes: %d\n", idx, len(eyeRects))
				if len(eyeRects) < 2 {
					faceMat.Close()
					continue
				}
				faceMat.Close()
				rects = append(rects, r)
			}
			if len(rects) > 0 {
				for _, r := range rects {
					gocv.Rectangle(&img, r, clr, 2)
				}
			}
		}

		window.IMShow(img)
		if window.WaitKey(80) & 0xFF == 'q' {
			fmt.Println("exit")
			break
		}
	}
}
