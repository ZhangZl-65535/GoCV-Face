package main

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"strings"
	"time"
	"gocv.io/x/gocv"
	"gocv.io/x/gocv/contrib"
	"./util"
)

type FaceInfo struct {
	Face      gocv.Mat // 经过 hash.Compute 之后的数据
	TimeStamp int64    // 拍到人脸时间戳
	TempItem  bool     // 临时数据，程序启动后通过摄像头新拍到的人脸
}
type RectInfo struct {
	Rect image.Rectangle
	Sim  int
}

var (
	m_mapFace  = util.NewSafeMap()
	m_oHash    = contrib.NewRadialVarianceHash()
	m_nSimilar = 60 // 人脸相似度阈值
)

func main() {
	ticker := time.NewTicker(time.Minute) // 每隔一分钟清除一次数据
	go func() {
		for {
			<-ticker.C
			clearCache(false)
		}
	}()
	defer ticker.Stop()

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

	loadData(faceclassifier, "./data/")

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
			timestamp := time.Now().UnixNano()
			count := 0
			rects := []RectInfo{}
			for idx, r := range faceRects {
				faceMat := img.Region(r)
				// detect eyes
				eyeRects := eyeclassifier.DetectMultiScale(faceMat)
				fmt.Printf("face %d: eyes: %d\n", idx, len(eyeRects))
				if len(eyeRects) < 2 {
					faceMat.Close()
					continue
				}
				sim, saved := saveFace(faceMat, idx, timestamp)
				faceMat.Close()
				if saved {
					count++
				}
				ri := RectInfo{
					Rect: r,
					Sim:  sim,
				}
				rects = append(rects, ri)
			}
			for _, rect := range rects {
				gocv.Rectangle(&img, rect.Rect, clr, 2)
				// draw text
				str := fmt.Sprintf("S: %d", rect.Sim)
				size := gocv.GetTextSize(str, gocv.FontHersheyPlain, 1, 2)
				gocv.PutText(&img, str, image.Pt(rect.Rect.Max.X - size.X, rect.Rect.Min.Y - 2), gocv.FontHersheyPlain, 1, clr, 2)
			}
			if count > 0 {
				saveFile(img, "./img/%s_%s.png", time.Unix(0, timestamp).Format("20060102_15-04-05"), getMillisecond(timestamp))
			}
		}

		window.IMShow(img)
		if window.WaitKey(10) & 0xFF == 'q' { // 按 Q 退出
			fmt.Println("exit")
			clearCache(true)
			break
		}
	}
}

func saveFace(src gocv.Mat, idx int, timestamp int64) (int, bool) {
	similar := 0
	cmptFace := gocv.NewMat()
	m_oHash.Compute(src, &cmptFace)
	for k, v := range m_mapFace.Items() {
		faceInfo := v.(*FaceInfo)
		if faceInfo == nil {
			continue
		}
		sim := compareFace(faceInfo.Face, cmptFace)
		if sim > m_nSimilar { // 替换数据
			fmt.Printf("--------Similar: %d face exist, file: %s\n", sim, k)
			if faceInfo.TempItem { // 非永久数据，替换为新数据
				replaceCache(faceInfo, cmptFace, timestamp)
			} else { // 永久数据，删除新数据
				cmptFace.Close()
			}
			return sim, false
		}
		if sim > similar {
			similar = sim
		}
	}
	filename := fmt.Sprintf("./img/%s_%s_%d.png", time.Unix(0, timestamp).Format("20060102_15-04-05"), getMillisecond(timestamp), idx)
	addCache(cmptFace, filename, timestamp, true)
	saveFile(src, filename)
	return similar, true
}
func saveFile(mat gocv.Mat, format string, args ...interface{}) {
	filename := fmt.Sprintf(format, args...)
	gocv.IMWrite(filename, mat)
}
func getMillisecond(timestamp int64) string {
	ms := fmt.Sprintf("%d", (timestamp / 1000000) % 1000)
	if len(ms) == 1 {
		ms = "00" + ms
	} else if len(ms) == 2 {
		ms = "0" + ms
	}
	return ms
}
// 加载指定目录下图片检测人脸
func loadData(faceclassifier gocv.CascadeClassifier, dirPath string) {
	dirhandle, err := os.Open(dirPath)
	if err != nil {
		return
	}
	defer dirhandle.Close()

	fis, err := dirhandle.Readdir(0)
	if err != nil {
		return
	}

	for _, fi := range fis {
		if fi.IsDir() {
			continue
		}
		if !strings.HasSuffix(fi.Name(), ".jpg") && !strings.HasSuffix(fi.Name(), ".png") && !strings.HasSuffix(fi.Name(), ".jpeg") {
			continue
		}
		mat := gocv.IMRead(dirPath + fi.Name(), gocv.IMReadGrayScale)
		rects := faceclassifier.DetectMultiScale(mat)
		if len(rects) == 0 {
			mat.Close()
			continue
		}
		face := mat.Region(rects[0])
		cmptFace := gocv.NewMat()
		m_oHash.Compute(face, &cmptFace)
		face.Close()
		addCache(cmptFace, fi.Name(), 0, false)
		mat.Close()
	}
}
func addCache(cmptFace gocv.Mat, filename string, timestamp int64, tempitem bool) {
	m_mapFace.Set(filename, &FaceInfo{
		Face:      cmptFace,
		TimeStamp: timestamp,
		TempItem:  tempitem,
	})
}
func replaceCache(faceInfo *FaceInfo, cmptFace gocv.Mat, timestamp int64) {
	faceInfo.TimeStamp = timestamp
	faceInfo.Face.Close()
	faceInfo.Face = cmptFace
}
func clearCache(force bool) {
	fmt.Println("[[[[clear cache data]]]]")
	if force {
		for k, v := range m_mapFace.Items() {
			item := v.(*FaceInfo)
			if item == nil {
				m_mapFace.Delete(k)
			} else {
				item.Face.Close()
				m_mapFace.Delete(k)
			}
		}
	}
	if m_mapFace.Length() < 100 { // 100张人脸
		return
	}
	timestamp := time.Now().Unix() - 300 // 5分钟前
	for k, v := range m_mapFace.Items() {
		item := v.(*FaceInfo)
		if item == nil {
			m_mapFace.Delete(k)
		} else if item.TempItem && item.TimeStamp < timestamp {
			item.Face.Close()
			m_mapFace.Delete(k)
		}
	}
}
func compareFace(face1, face2 gocv.Mat) int {
	sim := m_oHash.Compare(face1, face2)
	return int(sim * 100)
}
