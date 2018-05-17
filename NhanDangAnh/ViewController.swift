//
//  ViewController.swift
//  NhanDangAnh
//
//  Created by NguyenDucToan on 5/17/18.
//  Copyright © 2018 NguyenDucToan. All rights reserved.
//

import UIKit
import Vision
import CoreML

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    let mlModel = NameModel()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        importButton.addTarget(self, action: #selector(importFromCameraRoll), for: .touchUpInside)
        self.view.addSubview(previewImg)
        self.view.addSubview(descriptionLbl)
        self.view.addSubview(importButton)
    }

    func  AI(img :UIImage){
        
        
    }
    
    var importButton:UIButton = {
        let btn = UIButton(type: .system)
        btn.setTitle("Import", for: .normal)
        btn.setTitleColor(.white, for: .normal)
        btn.backgroundColor = .black
        btn.frame = CGRect(x: 0, y: 0, width: 110, height: 60)
        btn.center = CGPoint(x: UIScreen.main.bounds.width/2, y: UIScreen.main.bounds.height*0.90)
        btn.layer.cornerRadius = btn.bounds.height/2
        btn.tag = 0
        return btn
    }()
    
    var previewImg:UIImageView = {
        let img = UIImageView()
        img.frame = CGRect(x: 0, y: 0, width: 350, height: 350)
        img.contentMode = .scaleAspectFit
        img.center = CGPoint(x: UIScreen.main.bounds.width/2, y: UIScreen.main.bounds.height/3)
        return img
    }()
    
    var descriptionLbl:UILabel = {
        let lbl = UILabel()
        lbl.text = "No Image Content"
        lbl.frame = CGRect(x: 0, y: 0, width: 350, height: 50)
        lbl.textColor = .black
        lbl.textAlignment = .center
        lbl.center = CGPoint(x: UIScreen.main.bounds.width/2, y: UIScreen.main.bounds.height/1.5)
        return lbl
    }()

    @objc func importFromCameraRoll() {
        if UIImagePickerController.isSourceTypeAvailable(.photoLibrary) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = .photoLibrary;
            imagePicker.allowsEditing = true
            self.present(imagePicker, animated: true, completion: nil)
        }
    }
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        if let image = info[UIImagePickerControllerOriginalImage] as? UIImage {
            previewImg.image = image
            if let buffer = image.buffer(with: CGSize(width:224, height:224)) {
                guard let prediction = try? mlModel.prediction(image: buffer) else {fatalError("Unexpected runtime error")}
                descriptionLbl.text = prediction.personType
                print(prediction.personTypeProbability)
            }else{
                print("failed buffer")
            }
        }
        dismiss(animated:true, completion: nil)
    }
}
extension UIImage {
    func buffer(with size:CGSize) -> CVPixelBuffer? {
        if let image = self.cgImage {
            let frameSize = size
            var pixelBuffer:CVPixelBuffer? = nil
            let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(frameSize.width), Int(frameSize.height), kCVPixelFormatType_32BGRA , nil, &pixelBuffer)
            if status != kCVReturnSuccess {
                return nil
            }
            CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags.init(rawValue: 0))
            let data = CVPixelBufferGetBaseAddress(pixelBuffer!)
            let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue)
            let context = CGContext(data: data, width: Int(frameSize.width), height: Int(frameSize.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: bitmapInfo.rawValue)
            context?.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
            
            return pixelBuffer
        }else{
            return nil
        }
    }
}

