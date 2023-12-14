import { Injectable } from '@angular/core';

import { HttpClient, HttpParams, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { map, catchError} from 'rxjs/operators';


@Injectable()
export class FlaskapiService {

  //baseURL: string = "http://chat-birdt-backend:5000/";
  //baseURL: string = "http://172.27.0.2:5000/";
  //baseURL: string = "http://empty-backend:5000/";
  //baseURL: string = "backend/";

  constructor(private http: HttpClient) {
  }

  blank(address: string): Observable<any> {

    //return this.http.get(this.baseURL + 'question/' + text);
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });
    return this.http.get(address, { headers });
  }


}
