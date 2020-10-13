package com.example.photodatabase;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

public class MainActivity extends AppCompatActivity {

    SQLiteDatabase db=null;

    ImageView imageView = null;
    Button capture  = null;
    EditText tag = null;
    EditText size = null;
    ArrayList<Bitmap> resultImages;
    int currentImage = 0;
    TextView imageno= null;

    protected Interpreter tflite;
    private MappedByteBuffer tfliteModel;
    private TensorImage inputImageBuffer;
    private  int imageSizeX;
    private  int imageSizeY;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    private List<String> labels;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.iv);
        tag = (EditText) findViewById(R.id.tag);
        size = (EditText) findViewById(R.id.size);
        resultImages=new ArrayList<Bitmap>();
        currentImage=0;
        capture = (Button) findViewById(R.id.capture);
        imageno = (TextView) findViewById(R.id.imageno);

        db = this.openOrCreateDatabase ("PhotoDB", Context.MODE_PRIVATE, null);
        db.execSQL("drop table if exists Photo;");
        db.execSQL("create table if not exists Photo(id integer primary key autoincrement, tag text, size number, image blob);");

        try{
            tflite=new Interpreter(loadmodelfile(this));
        }catch (Exception e) {
            e.printStackTrace();
        }
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

        int probabilityTensorIndex = 0;
        int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        inputImageBuffer = new TensorImage(imageDataType);
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
    }

    public void takeSnap(View view) {
        Intent w=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(w,1);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Log.i("Lab1", "resultCode: " + resultCode+"requestCode: "+requestCode);
        if(requestCode==1 && resultCode==RESULT_OK){
            Bundle extras=data.getExtras();
            Bitmap imageBitmap=(Bitmap) extras.get("data");
            imageView.setImageBitmap(imageBitmap);
            Log.v("mytag","Bytes :"+imageBitmap.getRowBytes()*imageBitmap.getHeight());
            size.setText(String.valueOf(imageBitmap.getByteCount()));
            imageno.setText(null);

            inputImageBuffer = loadImage(imageBitmap);
            tflite.run(inputImageBuffer.getBuffer(),outputProbabilityBuffer.getBuffer().rewind());
            showresult();
        }
    }

    public void savePhoto(View view) {
        String tt = tag.getText().toString().trim();
        String s = size.getText().toString().trim();
        if(s.equals("") && tt.equals("")){
            return;
        }

        Bitmap bitmap=((BitmapDrawable) imageView.getDrawable()).getBitmap();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, bos);
        byte[] bArray = bos.toByteArray();

        if(tt.equals("")){
            ContentValues cv = new ContentValues();
            cv.put("size", s);
            cv.put("image",bArray);
            db.insert("Photo",null,cv);
        }
        else {
            StringTokenizer tokenizer = new StringTokenizer(tt, ";");
            while (tokenizer.hasMoreTokens()) {
                String t = tokenizer.nextToken().trim();
                ContentValues cv = new ContentValues();
                cv.put("tag", t);
                if (!s.equals("")){
                    cv.put("size", s);
                    Log.v("mytag","Size not null");
                }
                cv.put("image", bArray);
                db.insert("Photo", null, cv);
            }
        }
        printDB();
    }

    public void loadPhotos(View view) {
        resultImages.clear();
        String tt = tag.getText().toString().trim();
        String s = size.getText().toString().trim();
        if(s.equals("") && tt.equals("")){
            imageView.setImageBitmap(null);
            imageno.setText(null);
            return;
        }
        String s1="",s2="";
        if(!tt.equals("")){
            StringTokenizer tokenizer=new StringTokenizer(tt,";");
            String t;
            t=tokenizer.nextToken().trim();
            s1 += " (tag = '"+t+"'";
            while(tokenizer.hasMoreTokens()){
                t=tokenizer.nextToken().trim();
                s1 += " OR tag = '"+t+"'";
            }
            s1+=")";
        }
        if(!s.equals("")){
            float num= Float.parseFloat(s);
            s2+= " (size >= "+num*3/4+ " AND size <= "+num*5/4+ ")";
        }
        String sq = "SELECT DISTINCT image FROM Photo WHERE";
        if(s.equals("")) sq+=s1;
        else if(tt.equals("")) sq+=s2;
        else sq += s1+" AND"+s2;
        sq += ";";
        Log.v("mytag",sq);
        Cursor c = db.rawQuery(sq, null);
        Log.v("mytag","Total selected: "+c.getCount());
        c.moveToFirst();
        for(int i=0;i<c.getCount();i++){
            //Log.v("mytag", "id :" + c.getInt(0) + " tag :" + c.getString(1) + " size :" + c.getFloat(2));
            byte[] ba = c.getBlob(0);
            Bitmap b = BitmapFactory.decodeByteArray(ba, 0, ba.length);
            resultImages.add(b);
            c.moveToNext();
        }
        currentImage=0;
        if(resultImages.size()==0){
            imageView.setImageBitmap(null);
            imageno.setText(null);
        }
        else{
            imageView.setImageBitmap(resultImages.get(currentImage));
            imageno.setText((currentImage+1)+"/"+resultImages.size());
        }
    }



    public void printDB(){
        Log.v("mytag","-----Photo Table-----");
        Cursor c = db.rawQuery("SELECT * from Photo;", null);
        Log.v("mytag","Total rows: "+c.getCount());
        c.moveToFirst();
        for(int i=0;i<c.getCount();i++){
            Log.v("mytag","id :"+c.getInt(0)+" tag :"+c.getString(1)+" size :"+c.getFloat(2));
            c.moveToNext();
        }
    }

    public void moveRight(View view) {
        if(resultImages.size()>0 && currentImage<resultImages.size()-1){
            currentImage++;
            imageView.setImageBitmap(resultImages.get(currentImage));
            imageno.setText((currentImage+1)+"/"+resultImages.size());
        }
    }

    public void moveLeft(View view) {
        if(resultImages.size()>0 && currentImage>0){
            currentImage--;
            imageView.setImageBitmap(resultImages.get(currentImage));
            imageno.setText((currentImage+1)+"/"+resultImages.size());
        }
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor=activity.getAssets().openFd("mobilenet_v1_1.0_224_quant.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    private void showresult(){

        try{
            labels = FileUtil.loadLabels(this,"labels_mobilenet_quant_v1_224.txt");
        }catch (Exception e){
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        //float maxValueInMap =(Collections.max(labeledProbability.values()));

        List<Map.Entry<String, Float> > list =
                new LinkedList<Map.Entry<String, Float> >(labeledProbability.entrySet());
        // Sort the list
        Collections.sort(list, new Comparator<Map.Entry<String, Float> >() {
            public int compare(Map.Entry<String, Float> o1,
                               Map.Entry<String, Float> o2)
            {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });
        String res="";
        for(int i=0;i<5;i++){
            if(list.get(i).getValue()>0){
                res += list.get(i).getKey()+";";
                Log.v("mytag","key :"+list.get(i).getKey()+" Probability :"+list.get(i).getValue());
            }
        }
        tag.setText(res);
    }
}